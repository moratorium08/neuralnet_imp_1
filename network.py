# coding:utf-8
import pickle
import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
from chainer import serializers


GPU_FLAG = False
epoch = 20
rnn_size = 1000
batch_size = 32

class RNN(Chain):

    def __init__(self, n_v, n_u, train=True):
        super(RNN, self).__init__(
            embed=L.EmbedID(n_v, n_u),
            l1=L.LSTM(n_u, n_u),
            l2=L.LSTM(n_u, n_u),
            l3=L.Linear(n_u, n_v),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y


def load_dataset():
    with open("dataset", "r") as f1, open("vocaburary", "rb") as f2:
        # ごめんなさい闇です。
        dataset = f1.read().replace("2252,", "").replace("2256,", "").split(",")
        dataset = [int(x) for x in dataset]
        vocab = pickle.load(f2)
        l = np.array(dataset, np.int32)
        inv_vocab={}
        for x in vocab:
            inv_vocab[vocab[x]] = x
    return l[:40000], l[40000:50000], l[50000:], vocab, inv_vocab

train, test, valid, vocab, inb_vocab = load_dataset()

rnn = RNN(len(vocab), rnn_size)
model = L.Classifier(rnn)

if GPU_FLAG:
    cuda.init()
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)

# TrainerとUpdaterで代替可能だが
# 自分で実装してみた
t_size = len(train) - 1  # 最後のtrain分あぶないので
for e in range(epoch):
    print("{0}/{1}".format(e, epoch))
    loss = 0
    count = 0
    for i in range(t_size // batch_size):
        fd = i*batch_size
        bk = (i+1)*batch_size
        x_batch = train[fd:bk]
        y_batch = train[fd+1:bk+1]

        if GPU_FLAG:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)
        loss += model(x_batch, y_batch)
        count += 1
        # 途中で計算して捨てる
        if count % 10 == 0 or count == t_size//batch_size:
            print("truncate", loss.data)
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            loss = 0

serializers.save_hdf5("mymodel.h5", model)
#serializers.load_hdf5("mymodel.h5", model)

rnn.train = False
rnn.reset_state()
nxt = vocab["自己"]
nxt = 2000
while True:
    rnn.reset_state()
    nxt = int(input())
    for i in range(10):
        nxt = np.array([nxt], np.int32)
        prob = F.softmax(model.predictor(nxt))
        print(prob.data)
        nxt = np.argmax(prob.data)
        print(nxt)
