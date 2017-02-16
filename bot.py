# coding:utf-8
import yaml
from tweepy import OAuthHandler, API
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from network import RNN, load_dataset, rnn_size
import numpy as np
import time

keys = yaml.load(open("twitter_keys", "r"))
consumer_key = keys["consumer_key"]
consumer_secret = keys["consumer_secret"]
access_token = keys["access_token"]
access_secret = keys["access_secret"]

handler = OAuthHandler(consumer_key, consumer_secret)
handler.set_access_token(access_token, access_secret)

a, b, c, d, inv_vocab = load_dataset()

mx = len(inv_vocab)

api = API(handler)
rnn = RNN(mx, rnn_size, False)
model = L.Classifier(rnn)
serializers.load_hdf5("mymodel.h5", model)

while True:
    nxt = np.random.randint(0, mx)
    result = ""
    for i in range(40):
        nxt = np.array([nxt], np.int32)
        prob = F.softmax(model.predictor(nxt))
        nxt = np.argmax(prob.data)
        s = inv_vocab[nxt]
        if s == "。":
            break
        result += s

    print(result)
    try:
        api.update_status(result)
    except:
        print("error"
           7  rnn.reset_state()
    time.sleep(10)
