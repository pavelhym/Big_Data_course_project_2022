import pandas as pd
import glob
import os
import numpy as np
import copy

import matplotlib.dates as mdates
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
data = pd.read_csv('D://Documents//ITMO//Year1//Course_project_github//Big_Data_course_project_2022//storage//GME_only_values.csv').drop('Unnamed: 0', axis = 1)
data["date"] =  pd.to_datetime(data["date"])



stock = data['Adj Close']
date = data['date']




plt.figure(figsize=(12,6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=150))
plt.xticks(size=18)
plt.yticks(size=18)
plt.plot(date, stock, linewidth = 2)
plt.ylabel('Value', size=20)
plt.xlabel('Date', size=20)
plt.grid(axis='both')
plt.legend(fontsize=18)
plt.savefig('D://Documents//ITMO//Year1//Course_project_github//Big_Data_course_project_2022//plots//stock.png',dpi=300, format='png')
plt.show()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
stock_scaled = scaler.fit_transform(np.array(stock).reshape(-1,1))

comments_scaled = scaler.fit_transform(np.array(data['comments_num']).reshape(-1,1))


plt.figure(figsize=(12,6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=150))
plt.xticks(size=18)
plt.yticks(size=18)
plt.plot(date, stock_scaled, linewidth = 2, label = 'stock_price')
plt.plot(date, comments_scaled, linewidth = 2,  alpha=0.5, label = 'comments_num')
plt.ylabel('Value', size=20)
plt.xlabel('Date', size=20)
plt.gca().axes.yaxis.set_ticklabels([])
plt.legend(fontsize=18)
plt.grid(axis='both')
plt.savefig('plots/stock_comments.png',dpi=300, format='png')
plt.show()



data1 = pd.read_csv('final_data/data_f0_t300.csv')


import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('words')
nltk.download('stopwords')

stemmer = nltk.PorterStemmer()
lemm = nltk.WordNetLemmatizer()

eng_words = set(nltk.corpus.words.words())

stop_words = set(stopwords.words('english'))

import re
cleaned_data=[]
a = 0 
for i in ['yolo']:
    print(a, end = '-')
    a+=1
    if i != "":
        tweet = re.sub(r"\/r|\/n|\/t", '', i)
        tweet = re.sub(r"\\n|\\t|\\r", '', tweet)
        tweet=re.sub('[^a-zA-Z]',' ', tweet)
        tweet = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", ' ', tweet)
        tweet = str(tweet).replace("\n", '')
        tweet = str(tweet).replace("/r", '')
        tweet = str(tweet).replace("[removed]", '')
 
        tweet=tweet.lower().split()
        tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words)]
        # tweet=[lemm.lemmatize(word) for word in tweet if (word not in stop_words)]
 
        tweet=' '.join(tweet)
        cleaned_data.append(tweet)
    else:
        tweet=' '
        cleaned_data.append(tweet)


