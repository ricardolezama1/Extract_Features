#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import sys
#initialize TF IDF vector object
vectorizer = TfidfVectorizer()

#input file 
language_data = open(sys.argv[0], 'r')

Vectora = vectorizer.fit_transform(language_data)
a = vectorizer.get_feature_names()

#list nltk stopwords,custom and list concatenation helps for adding more stopwords. 
stop1 = stopwords.words('english')
stop2 = stopwords.words('spanish')
#custom stopword list
custom = ["クロス"]


stop_combined = stop1 + stop2 + custom


document = [i for i in a if i not in stop_combined]

#evaluate 
print (document)
