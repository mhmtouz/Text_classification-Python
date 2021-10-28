# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:37:26 2019

@author: black
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import string
import unicodedata
import sys
# Kullanılan farklı noktalamayı tutmak için bir tablo yapısı
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))


# Noktalama işaretlerini cümlelerden kaldırma yöntemi.
def remove_punctuation(text):
    return text.translate(tbl)

#stemmer'ı başlat
stemmer = LancasterStemmer()
# Json verilerini dosyadan okumak için # değişken
data = None

# json dosyasını oku ve egzersiz verilerini yükle
with open('data.json',encoding="utf8") as json_data:
    data = json.load(json_data)
    print(data)

# eğitmek için tüm kategorilerin bir listesini al
categories = list(data.keys())
words = []
#  cümle ve kategori ismindeki kelimelerin yer aldığı bir liste
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # cümleyi noktalama işaretlerini kaldır
        each_sentence = remove_punctuation(each_sentence)
        print(each_sentence)
        # her cümleden kelime ayıkla ve kelime listesine ekle
        w = nltk.word_tokenize(each_sentence)
        print("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))

# Her kelimeyi ÇIKAR ve indirin ve kopyaları kaldırın
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print(words)
print(docs)

# eğitim verilerimizi oluştur
training = []
output = []
# çıktılarımız için boş bir dizi oluşturun
output_empty = [0] * len(categories)


for doc in docs:
    # listedeki her bir belge için kelimeler çantamızı (yay) başlatalım
    bow = []
    #  kalıp için tokenize edilmiş kelimelerin listesi
    token_words = doc[0]
    # her kelimeyi ÇIKAR
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # kelime dizisi oluştur
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # Eğitim setimiz, bir kelime torbası modeli ve çıktı satırını içerecektir.
    # Hangi yayın ait olduğu catefory.
    training.append([bow, output_row])

#bizim özelliklerimizi karıştırır ve tensorflow numpy dizisini alırken np.array işlevine dönüşür
random.shuffle(training)
training = np.array(training)

# trainX kelimelerin Torbasını ve train_y etiketi / kategorisini içerir
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# temel grafik verilerini sıfırla
tf.reset_default_graph()
#  Yapay sinir ağı oluştur
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Model ve kurulum tensorboard tanımlayın
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Eğitimi başlat (degrade iniş algoritması uygula)
model.fit(train_x, train_y, n_epoch=10000, batch_size=32, show_metric=True)
model.save('model.tflearn')

# mdodel'i birkaç cümle için test edelim:
# İlk iki cümle eğitim için kullanılan ve son iki cümle eğitim verileri mevcut değildir.
sent_1 = "saat kaç?"
sent_2 = "şimdi gitmem gerekiyor"
sent_3 = "Ben derste heyecanlandım"

# cümle ve tüm kelimelerin listesini alan bir yöntem
# ve tensorflow'a beslenebilecek formdaki verileri döndürür

def get_tf_record(sentence):
    global words
    #  deseni tokenize et
    sentence_words = nltk.word_tokenize(sentence)
    # her kelimeyi ekle
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # kelimelerin dizisi
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


# 4 cümlenin sonuçlarını tahmin etmeye başlayabiliriz
print(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_2)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent_3)]))])
