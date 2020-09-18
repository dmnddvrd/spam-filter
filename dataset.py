import os
import random
import re
import tarfile
import zipfile

stopwords = set(open('stopwords.txt').read().lower().split())
stopwords |= set(open('stopwords2.txt').read().lower().split())


def get_words(raw):
    text = raw.decode('latin-1').lower()[len('Subject:'):]
    return [x for x in re.findall(r'[a-z]{2,}', text) if x not in stopwords]


def load():
    tar = tarfile.open('enron6.tar.gz')

    dataset = {}

    for fd in tar:
        buf = tar.extractfile(fd)
        match = re.match(r'^enron6/(spam|ham)/', fd.path)
        if match:
            label = match.group(1)
            filename = os.path.basename(fd.path)
            dataset[filename] = get_words(buf.read()), label

    train = [dataset[x] for x in open('train.txt').read().split()]
    test = [dataset[x] for x in open('test.txt').read().split()]

    random.shuffle(train)
    random.shuffle(test)

    return [
        [words for words, label in train],
        [label == 'spam' for words, label in train],
        [words for words, label in test],
        [label == 'spam' for words, label in test],
    ]


def ssl():
    data = []
    pack = zipfile.ZipFile('ssl.zip')
    for name in pack.namelist():
        data.append(get_words(pack.read(name)))

    random.shuffle(data)
    return data
