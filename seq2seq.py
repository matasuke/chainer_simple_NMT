import argparse
import numpy as np
import chainer
import pickle
from chainer.functions as F
from chainer.links as L
from chainer import backends

def convert2numpy(sentences):
    sentences_np = np.array([np.array(sen['encoded_sentence'], dtpye=int32) for sen in sentences])
    return sentences_np

def load_pickle(in_file):
    with open(in_file, 'rb') as f:
        row_data = pickle.load(f)

    return row_data

def calc_UNK(sentences, word_ids):
    #numpy array
    unk = sum((s == word_ids['<UNK>']).sum() for s in sentences)
    total = sum(s.size for s in sentences)

    return unk / total

class seq2seq(chainer.Chain):
    def __init__(self, n_layers, in_vocab, out_vocab, n_mid):
        super(seq2seq, self).__init__()
        with self.init_scope():
            self.embed_in = L.EmbedID(in_vocab, n_mid)
            self.embed_out = L.EmbedID(n_mid, out_vocab)
            self.encoder = L.NStempLSTM(n_layers, n_mid, n_mid, 0.1)
            self.decoder = L.NStemLSTM(n_layers, n_mid, n_mid, 0.1)
            self.l1 = L.Linear(n_mid, n_mid)

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]


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
