import collections
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import training

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
                h, c, ys = self.decoder(h, c, eys)
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
