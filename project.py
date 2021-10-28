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
import gui
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from nltk.tree import *

 
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


tf.reset_default_graph()
#  Yapay sinir ağı oluştur
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Model ve kurulum tensorboard tanımlayın
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('model.tflearn')

# mdodel'i birkaç cümle için test edelim:
# İlk iki cümle eğitim için kullanılan ve son iki cümle eğitim verileri mevcut değildir.
sent1 = """Bu devreye kadar horoskobun alt yarısı, yani subjektif ve bilinçsiz kısım geçilmiştir. Birey bu alana gelinceye dek, kendine bakışında subjektif ve bilinçsizdir. Dolayısıyla, kendini diğerlerinin gördüğü gözle görmesi imkansızdır.

İlk yarı boyunca, doğum evvelinden taşınan yaşamsal iç güdüler maskelenmiş ve yok edilmeye çalışılmıştır. Doğumda oluşan kişilik çekirdeği, alt yarımkürede dışardan gelen ve ebeveynlerden kaynaklanan etkiler doğrultusunda, ancak nereye varacağı, nereye götüreceği belirsiz bir durumda şekillenmiştir.

Genç bu devrede, toplumun ve ebeveynlerinin kendisine empoze ettiği özellikler doğrultusunda hayatını yönlendirir. Bu aşamada, objektivitenin başlangıcı olarak kabul edilecek gerçek bilinçlilik halinin, alternatif görüşlerin ve fikirlerin tanınması, benimsenmesi söz konusudur. Dolayısıyla, bireyin görünen ile görünmeyen yüzü ya da, gerçek varlığı arasında bir denge oluşturma aşamasına gelinir."""
# cümleyi noktalama işaretlerini kaldır

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
            eq=0.0
            count=0
            j=0
            while(j<len(w) and j<len(s)):
                if w[j]==s[j]:
                    count+=1
                else:
                    break
                j+=1
            if len(w)>len(s):
                eq=count/len(w)
            else:
                eq=count/len(s)
            
            if eq>0.5:
                bow[i] = eq
    return(np.array(bow))


# 4 cümlenin sonuçlarını tahmin etmeye başlayabiliriz
ozet=""
w = nltk.sent_tokenize(sent1)
anlam=[]
anlam1=[]
for i in range(len(w)):
    print(w[i])
    indis= np.argmax(model.predict([get_tf_record(w[i])]))
    print(categories[indis])
    anlam.append(categories[indis])
    anlam1.append(categories[indis])


anlam_indis=[]
anlam_sayisi=[]
a=0
while(len(anlam1)>0):
    anlam_indis.append(anlam1[0])
    anlam_sayisi.append(0)
    i=0
    for i in range(len(anlam1)):
        if anlam_indis[a] == anlam1[i]:
            anlam_sayisi[a]+=1
    
    for j in range(anlam_sayisi[a]):
            anlam1.remove(anlam_indis[a])
    a+=1
print (anlam_indis)
print (anlam_sayisi)
enb=0
enb2=0
temp=0
print(len(anlam),len(w))
for i in range(len(anlam_sayisi)):
    if anlam_sayisi[i]>enb:
        enb2=enb
        enb=anlam_sayisi[i]
    elif anlam_sayisi[i]>enb2:
        enb2=anlam_sayisi[i]
enb_indis=anlam_sayisi.index(enb)
enb2_indis=anlam_sayisi.index(enb2)
print(anlam_indis[enb_indis])
print(anlam_indis[enb2_indis])
yuzde_enb=enb*50/100
yuzde_enb2=enb2*10/100
say_enb=0
say_enb2=0
print(len(anlam),len(w))
for i in range(len(w)):
    if anlam[i]==anlam_indis[enb_indis] and say_enb<yuzde_enb:
        say_enb+=1
        ozet+=w[i]
    elif anlam[i]==anlam_indis[enb2_indis] and say_enb2<yuzde_enb2:
        say_enb2+=1
        ozet+=w[i]
print("\n")
print(ozet)
