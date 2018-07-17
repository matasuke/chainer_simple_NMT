import argparse
import collections
import numpy as np
from nltk.translate import bleu_score
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import training
from chainer.training import extensions
from chainer import serializers

from Seq2SeqDataset import Seq2SeqDatasetBase
from common.record import record_settings
from common.make_dirs import create_save_dirs

tokens = collections.Counter({
    '<UNK>': 0,
    '<SOS>': 1,
    '<EOS>': 2
})


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_selection = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_selection, 0)

    return exs

class seq2seq(chainer.Chain):
    def __init__(
            self,
            n_layers,
            n_source_vocab,
            n_target_vocab,
            n_units,
            dropout_ratio=0.1
    ):

        super(seq2seq, self).__init__()
        self.n_layers = n_layers
        self.n_source_vocab = n_source_vocab
        self.n_target_vocab = n_target_vocab
        self.n_units = n_units

        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout_ratio)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, dropout_ratio)
            self.l1 = L.Linear(n_units, n_target_vocab)

    def __call__(self, xs, ys):
        batch = len(xs)
        # delete <SOS> and <EOS> from x and reverse the order
        xs = [x[1:-1][::-1] for x in xs]

        ys_in = [y[:-1] for y in ys]
        ys_out = [y[1:] for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.l1(concat_os), concat_ys_out, reduce="no")) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        prep = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'prep': prep}, self)

        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # delete <SOS> and <EOS> from x and reverse the order
            xs = [x[1:-1][::-1] for x in xs]

            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, tokens['<SOS>'], np.int32)

            result = []
            for _ in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, axis=0)
                _, _, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.l1(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                result.append(ys)

            result = cuda.to_cpu(self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

            outs = []
            for y in result:
                inds = np.argwhere(y == tokens['<EOS>'])
                if inds:
                    y = y[:inds[0, 0]]
                outs.append(y)

            return outs

class CalculateBleu(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self,
            model,
            test_data,
            key,
            batch=100,
            device=-1,
            max_length=100
    ):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, iterator):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references,
            hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1
        )
        chainer.report({self.key: bleu})

def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)

            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('SORCE', type=str,
                        help='preprocessed source data path')
    parser.add_argument('TARGET', type=str,
                        help='preprocessed target data path')
    parser.add_argument('--validation_sources', type=str, default='',
                        help='preprocessed validation sorce data path')
    parser.add_argument('--validation_targets', type=str, default='',
                        help="preprocessed validation target data path")
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help="numbe of sentence pairs in each mini-batch")
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help="number of epoch to train")
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help="GPU ID(negative value indicates CPU)")
    parser.add_argument('--reuse', '-r', type=str, default='',
                        help="reuse the training from snapshot")
    parser.add_argument('--unit', '-u', type=int, default=2048,
                        help="number of units")
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help="number of layers")
    parser.add_argument('--n_source_min_token', type=int, default=1,
                        help="number of min tokens in source sentences")
    parser.add_argument('--n_source_max_token', type=int, default=50,
                        help="number of max tokens in source sentences")
    parser.add_argument('--n_target_min_token', type=int, default=1,
                        help="number of min tokens in target sentences")
    parser.add_argument('--n_target_max_token', type=int, default=50,
                        help="number of max tokens in target sentences")
    parser.add_argument('--log_interval', type=int, default=200,
                        help="number of iteration to show log")
    parser.add_argument('--validation_interval', type=int, default=200,
                        help="number of iteration to evaluate the model")
    parser.add_argument('--snapshot_interval', type=int, default=200,
                        help='number of iteration to save model and optimizer')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help="directory to output the result")
    args = parser.parse_args()

    train_data = Seq2SeqDatasetBase(
        args.SORCE,
        args.TARGET,
        args.n_source_min_token,
        args.n_source_max_token,
        args.n_target_min_token,
        args.n_target_max_token
    )

    # make output dirs
    save_dirs = create_save_dirs(args.out)

    # print dataset configurations
    dataset_configurations = train_data.get_configurations
    for key, value in dataset_configurations.items():
        print(key + '\t' + value)

    # make configuration file and save it
    configurations = vars(args)
    record_settings(save_dirs['log_dir'], configurations, dataset_configurations)

    # setup model
    model = seq2seq(
        args.layer,
        len(train_data.get_source_word_ids),
        len(train_data.get_target_word_ids),
        args.unit,
        dropout_ratio=0.1
    )

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if args.reuse:
        serializers.load_npz(args.resume, model)

    # setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # setup updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    '''
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/prep',
             'validation/main/loss', 'validation/main/prep', 'elapsed_time']
        ),
        trigger=(args.log_interval, 'iteration')
    )
    '''
    trainer.extend(
        extensions.LogReport(
            ['epoch', 'iteration', 'main/loss', 'main/prep',
             'validation/main/loss', 'validation/main/prep', 'elapsed_time']
        ),
        trigger=(args.log_interval, 'iteration')
    )
    trainer.extend(extensions.ProgressBar())
    trainer.extend(
        extensions.snapshot(),
        trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(
            model,
            'model_iter_{.updater.iteration}'
        ),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(
        extensions.snapshot_object(
            optimizer,
            'optimizer_iter_{.updater.iteration}'
        ),
        trigger=(args.snapshot_interval, 'iteration')
    )

    if args.validation_sources and args.validation_targets:
        test_data = Seq2SeqDatasetBase(
            args.validation_sources,
            args.validation_targets,
            args.n_source_min_token,
            args.n_source_max_token,
            args.n_target_min_token,
            args.n_target_max_token,
            validation=True
        )

        print('validation data size: %d' % len(test_data))
        print('Source unk ratio: %.2f%%' % test_data.source_unk_ratio)
        print('Target unk ratio: %.2f%%' % test_data.target_unk_ratio)

        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[np.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join(test_data.source_index2token(source)[1:-1])
            target_sentence = ' '.join(test_data.target_index2token(target)[1:-1])
            result_sentence = ' '.join(test_data.target_index2token(result)[1:-1])
            print('# source : ' + source_sentence)
            print('# target : ' + target_sentence)
            print('# result : ' + result_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, 'validation/main/bleu', device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

    print('start training')
    trainer.run()

    serializers.save_npz(save_dirs['final_result'] / 'model_final', model)
    serializers.save_npz(save_dirs['final_result'] / 'optimizer_final', optimizer)


if __name__ == '__main__':
    main()
