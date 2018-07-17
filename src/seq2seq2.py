import argparse

import chainer
from chainer.backends import cuda
from chainer import training
from chainer.training import extensions
from chainer import serializers

from Seq2SeqDataset import Seq2SeqDatasetBase
from common.record import record_settings
from common.make_dirs import create_save_dirs
from common.convert import convert
from extensions.translation import Translation
from extensions.CalculateBleu import CalculateBleu
from net import seq2seq

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
        print(key + '\t' + str(value))

    # make configuration file and save it
    record_settings(save_dirs['log_dir'], args, dataset_configurations)

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
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/prep',
             'validation/main/loss', 'validation/main/prep', 'elapsed_time']
        ),
        trigger=(args.log_interval, 'iteration')
    )
    trainer.extend(
        extensions.LogReport(
            ['epoch', 'iteration', 'main/loss', 'main/prep',
             'validation/main/loss', 'validation/main/prep', 'elapsed_time']
        ),
        trigger=(args.log_interval, 'iteration')
    )
    # trainer.extend(extensions.ProgressBar())
    trainer.extend(
        extensions.snapshot(),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(
        extensions.snapshot_object(
            model,
            filename='model_iter_{.updater.iteration}'
        ),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(
        extensions.snapshot_object(
            optimizer,
            filename='optimizer_iter_{.updater.iteration}'
        ),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            x_key='epoch',
            trigger=(1, 'epoch'),
            file_name='loss.png'
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/prep', 'validation/main/prep'],
            x_key='epoch',
            trigger=(1, 'epoch'),
            file_name='prep.png'
        )
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

        test_dataset_configurations = test_data.get_configurations
        for key, value in test_dataset_configurations.items():
            print(key + '\t' + value)

        trainer.extend(
            Translation(model, model.translate, test_data),
            trigger=(args.validation_interval, 'iteration')
        )
        trainer.extend(
            model.translate, trigger=(args.validation_interval, 'iteration'))
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
