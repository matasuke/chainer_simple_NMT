import argparse
import numpy as np
import chainer
import pickle

import chainer
from chainer.backends import cuda
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extentions

UNK = 0
EOS = 1

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_selection = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_selection, 0)

    return exs

class Seq2Seq(chainer.Chain):
    
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStemLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStempLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]
        
        eos = self.xp.array([EOS], np.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)

        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        prep = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'prep': prep}, self)
        
        return loss

    def translate(self, xs, max_length = 100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, np.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                result.append(ys)

        retult = cuda.to_cpu(
                self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

    def convert(batch, device):
        def to_device_batch(batch):
            if device is None:
                return batch
            elif device < 0:
                return [chainer.dataset.to_device(device, x) for x in batch]
            else:
                xp = cuda.cupy.get_array_module(*batch)
                concat = xp.concatenate(batch, axis=0)
                sections = np.cumsum([len(x) for x in batch[::-1]], dtype=np.int32)
                concat_dev = chainer.dataset.to_device(device, concat)
                batch_dev = cuda.cupy.split(concat_dev, sections)
                return batch_dev

        return {'xs': to_device_batch([x for x, _ in batch]),
                'ys': to_device_batch([y for _, y in batch])}




def load_pickle(in_file):
    with open(in_file, 'rb') as f:
        row_data = pickle.load(f)

    return row_data

def calc_UNK(sentences, word_ids):
    #numpy array
    unk = sum((s == word_ids['<UNK>']).sum() for s in sentences)
    total = sum(s.size for s in sentences)

    return unk / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_process', action="store_false")
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--source_val', type=str)
    parser.add_argument('--target_val', type=str)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n_mid', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=3)
    args = parser.parse_args()

    
    #add pre_process functions
    source_data = load_pickle(args.source)['train']
    target_data = load_pickle(args.target)['train']
    
    source_sentences = convert2numpy(source_data['sentences'])
    source_vocab = source_data['words_ids']
    target_sentences = convert2numpy(target_data['sentences'])
    target_vocab = target_data['words_ids']
    assert len(source_sentences) == len(target_sentences)

    # min and max token size should be designeted here
    train_data = [(s, t) for s, t in zip(source_sentences, target_setences)]

    source_UNK_= calc_UNK_ratio(source_sentences)
    target_UNK_= calc_UNK_ratio(target_sentences)

    print('source vocabrary size: %d' % len(source_vocab))
    print('target vocabrary size: %d' % len(target_vocab))
    print('training data size: %d' % len(train_data))
    print('source UNK ratio; %.2f%%' % source_UNK*100)
    print('target UNK ratio; %.2f%%' % target_UNK*100)
    
    
    #setup model
    model = seq2seq(args.layers, len(source_vocab), len(target_vocab), args.n_mid)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.Iterators.SerialIterator(train_data, args.batchsize)
