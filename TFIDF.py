# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:55:02 2023

@author: Ifra
"""

import pandas as pd
from pandas import DataFrame
from itertools import chain
import numpy as np
import collections
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import time
#encoding= 'unicode_escape'
#sep=';'

start_time = time.time()
file = "dataraw.csv"
data = pd.read_csv(file, sep=';')
berita = data.isi_berita
pisah = []

for bow in berita:
    print(bow)
    if bow=="none":
        pisah.append("none")  
    else:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        listStopword =  set(stopwords.words('indonesian'))
        #tokenization
        isiberita = ['com', 'id']
        # Remove alphanumeric
        review = re.sub(r'[^\w\s]', ' ', str(bow))
        # Remove @username
        review = re.sub('@[^\s]+', '', str(bow))
        # Remove #tagger
        review = re.sub(r'#([^\s]+)', '', str(bow))
        # Remove angka termasuk angka yang berada dalam string
        # Remove non ASCII chars
        review = re.sub(r'[^\x00-\x7f]', r'', str(bow))
        review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', str(bow))
        review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str(bow))
        review = re.sub(r'\\u\w\w\w\w', '', str(bow))
        # Remove simbol, angka dan karakter aneh
        review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", str(bow))
        for d in isiberita:
            if d in review:
                review = review.replace(d, ' ')     
        delete_number = ''.join([i for i in review  if not i.isdigit()])
        ubah_teks = ['dgn', 'sy', '&', 'dst', 'u/', 'jabar', 'bbrpa', 'tp', 'mnit', 'dsb', 'dll', 'slh', 'bs', 'sblumnya', 'kdepan', 'org', 'pelabuhan ratu', 'Pelabuhan Ratu']
        for p in ubah_teks:
            delete_number = delete_number.replace(p, 'dengan').replace(p, 'saya').replace(p, 'dan').replace(p, 'dan sekitarnya').replace(p, 'untuk').replace(p, 'jawa barat').replace(p, 'beberapa').replace(p, 'tapi').replace(p, 'menit').replace(p, 'dan lain-lain').replace(p, 'salah').replace(p, 'bisa').replace(p, 'sebelumnya').replace(p, 'ke depan').replace(p, 'orang').replace(p, 'palabuhanratu')
        tokens = word_tokenize(delete_number)
        print(tokens)
        hasil = []
        for t in tokens:
            #casefolding
            casefolding = t.lower()
            print(casefolding)
            removed = []
            #stopword
            if t not in listStopword:
                removed.append(casefolding)
                print(removed)
                #stemming
                for r in removed: 
                    hasil.append(stemmer.stem(r))
        pisah.append(hasil)
        
print(pisah)

DF = {}
N = len(pisah)
for i in range(N):
    tokens = pisah[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
            
for i in DF:
    DF[i] = len(DF[i])

print(DF)
total_vocab_size = len(DF)
total_vocab = [x for x in DF]
print(total_vocab)
print("\n bobot")
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

doc = 0
tf_idf = {}
bobot = []
for i in range(N):
    print(pisah[i])
    if pisah[i]=="none":
        bobot.append("none")
    else:
        tokens = pisah[i] 
        counter = collections.Counter(tokens)
        words_count = len(tokens)
        angka = []
        for token in DF:
            try:
                tf = counter[token]/words_count
            except ZeroDivisionError:
                tf = counter[token]
            df = doc_freq(token)
            idf = math.log10((N)/(df))
            
            tf_idf[token] = tf*idf
            
        total=sum(tf_idf.values())
        print(tf_idf)
        #print(total)
        bobot.append(total)  
    
data['bobot_isi_berita'] = bobot 
data.to_csv(file, index=False)
print("Data Tersimpan dalam file " + file )
print("--- %s seconds ---" % (time.time() - start_time))