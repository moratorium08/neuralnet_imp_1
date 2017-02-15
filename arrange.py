# coding:utf-8
from MeCab import Tagger
import codecs

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
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset.append(vocab[word])

for x in vocab:
    print(x)
print("vocaburary size:", len(vocab))
print("dataset size:", len(dataset))

with open("dataset", "w") as f:
    f.write(','.join(map(str, dataset)))
