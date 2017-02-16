# coding:utf-8
from MeCab import Tagger
import codecs
import pickle

tagger = Tagger("-Ochasen")

words = []

with codecs.open("tweets", "r") as f:
    tweets = f.read().replace("\n", "。")
    result = tagger.parseToNode(tweets)
    while result:
        # 眠いから、根本解決諦めた
        # unicodeバグるの死んでくれ
        try:
            words.append(result.surface)
        except:
            print("tsurai")
        result = result.next

    vocab = {}
    dataset = []
    for i, word in enumerate(words):
        if i == 0:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset.append(vocab[word])

print("vocaburary size:", len(vocab))
print("dataset size:", len(dataset))

with open("dataset", "w") as f:
    f.write(','.join(map(str, dataset)))
with open("vocaburary", "wb") as f:
    pickle.dump(vocab, f)
