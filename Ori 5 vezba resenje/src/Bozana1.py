import re
import string
import math
import pandas as pd
from pandas import DataFrame


def load_data():
    # TODO 1: ucitati podatke iz data/train.tsv datoteke
    # rezultat treba da budu dve liste, texts i sentiments

    train = pd.read_csv('../data/train.tsv', sep='\t')

    texts = train.text.values
    sentiments = train.sentiment.values

    # texts, sentiments = [], []
    # with open('train.tsv','r') as file
    #   read=csv.reader(codecs.iterdecode(file,'utf-8'),delimiter='\t\)
    #   for i,row in enumerate(read):
    #       if i=0: #skip header
    #           continue:
    #       sentiments.append(row[0])
    #       texts.append(row[1])

    return texts, sentiments


def preprocess(text):
    # TODO 2: implementirati preprocesiranje teksta
    # - izbacivanje znakova interpunkcije
    # - svodjenje celog teksta na mala slova
    # rezultat treba da bude preprocesiran tekst
    text = text.lower()
    text = re.sub('[^a-zA-Z]+', ' ', text).strip()
    return text


def tokenize(text):
    text = preprocess(text)
    # TODO 3: implementirati tokenizaciju teksta na reci
    # rezultat treba da bude lista reci koje se nalaze u datom tekstu
    words = text.split(' ')
    return words


def count_words(words):
    #words = tokenize(text)
    # TODO 4: implementirati prebrojavanje reci u datum tekstu
    # rezultat treba da bude mapa, ciji kljucevi su reci, a vrednosti broj ponavljanja te reci u datoj recenici

    words_count = {}
    for word in words:
        words_count[word] = words_count.get(word, 0.0) + 1.0

    return words_count


def fit(texts, sentiments):
    # inicijalizacija struktura
    bag_of_words = {}  # bag-of-words za sve recenzije
    words_count = {'pos': {},  # isto bag-of-words, ali posebno za pozivitne i negativne recenzije
                   'neg': {}}
    texts_count = {'pos': 0.0,  # broj tekstova za pozivitne i negativne recenzije
                   'neg': 0.0}

    # TODO 5: proci kroz sve recenzije i sentimente i napuniti gore inicijalizovane strukture
    # bag-of-words je mapa svih reci i broja njihovih ponavljanja u celom korpusu recenzija
    for text, sentiment in zip(texts, sentiments):
        texts_count[sentiment] += 1
        words = tokenize(text)
        counts = count_words(words)
        for word, count in list(counts.items()):
            if word not in bag_of_words:
                bag_of_words[word] = 0.0
            if word not in words_count[sentiment]:
                words_count[sentiment][word] = 0.0
            bag_of_words[word] += count
            words_count[sentiment][word] += count

    return bag_of_words, words_count, texts_count


def predict(text, bag_of_words, words_count, texts_count):
    words = tokenize(text)  # tokenizacija teksta

    # TODO 6: implementirati Naivni Bayes klasifikator za sentiment teksta (recenzije)
    # rezultat treba da bude mapa verovatnoca da je dati tekst klasifikovan kao pozitivnu i negativna recenzija

    counts = count_words(words)
    score_pos, score_neg = 0.0, 0.0

    prior_neg = (texts_count['neg'] / sum(texts_count.values()))
    prior_pos = (texts_count['pos'] / sum(texts_count.values()))

    log_prob_neg = 0.0
    log_prob_pos = 0.0
    for w, cnt in list(counts.items()):
        if len(w) <= 2:
            continue
        p_word = (bag_of_words.get(w, 0.0) + 1.0)/ sum(bag_of_words.values())
        p_w_given_pos = (words_count['pos'].get(w, 0.0) + 1) / (
                    sum(words_count['pos'].values()) + sum(bag_of_words.values()))
        p_w_given_neg = (words_count['neg'].get(w, 0.0) + 1) / (
                    sum(words_count['neg'].values()) + sum(bag_of_words.values()))

        if p_w_given_pos > 0:
            log_prob_pos += math.log(cnt + p_w_given_pos / p_word)
        if p_w_given_neg > 0:
            log_prob_neg += math.log(cnt + p_w_given_neg / p_word)

        score_pos = math.exp(log_prob_pos + math.log(prior_pos))
        score_neg = math.exp(log_prob_neg + math.log(prior_neg))

    return {'pos': score_pos, 'neg': score_neg}


if __name__ == '__main__':
    # ucitavanje data seta
    texts, sentiments = load_data()

    # izracunavanje / prebrojavanje stvari potrebnih za primenu Naivnog Bayesa
    bag_of_words, words_count, texts_count = fit(texts, sentiments)

    # recenzija
    text = 'I really liked this movie.'

    # klasifikovati sentiment recenzije koriscenjem Naivnog Bayes klasifikatora
    predictions = predict(text, bag_of_words, words_count, texts_count)

    print('-' * 30)
    print('Review: {0}'.format(text))
    print('Score(pos): {0}'.format(predictions['pos']))
    print('Score(neg): {0}'.format(predictions['neg']))
