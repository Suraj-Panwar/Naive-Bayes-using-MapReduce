
# coding: utf-8

# In[94]:


import nltk 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import string


# In[95]:


import gensim


# In[96]:

# Read Data
data = pd.read_csv('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\part.txt', sep ='\t', header= None)
data_dev = pd.read_table('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\dev.csv' , sep='\t<', header=None)


# In[97]:

# Simple preprocessing
data_dev[0] = [x[1:] for x in data_dev[0]]


# In[98]:


data_dev[0] = [x.split(',') for x in data_dev[0]]


# In[99]:

# Create Dictionary using Data
current_label = data[0][0]
dict_train = {}
for label in data[0].unique():
    temp_data = data[data[0] == label]
    dict_temp = {}
    for word, value in zip(temp_data[1], temp_data[2]):
        dict_temp[word] = value
    dict_train[label] = dict_temp


# In[100]:


dict_main = {}
for item, count in zip(data[1], data[2]):
    if item in dict_main:
        dict_main[item] += count
    else:
        dict_main[item] = count


# In[101]:

# Main program giving the accuracy
def find_class(main_dic, dic, sample):
    words = [word for word in gensim.utils.simple_preprocess(sample) if word not in stopwords][15:]

    dict_store = {key:0 for key in dic.keys()}
    length = len(main_dic)/10
    for word in words:
        for key in dic:
            if word not in dic[key]:
                dict_store[key] += -np.log(length)
            else:
                dict_store[key] += np.log((dic[key][word] + 1)/ (main_dic[word] + length))
    maxi = -9999999
    for key, value in zip(dict_store.keys(), dict_store.values()):
        if value > maxi:
            ret = key
            maxi = value
        #print(key)
    return ret


# In[102]:


score = 0
total = 0
for i, j in zip(range(len(data_dev)), data_dev[0]):
    total +=1 
    if total%1000 ==0:
        print('Accuracy: {0:1.2f}, Samples: {1:5.0f}'.format((score/total)*100, total))
    clas = find_class(dict_main, dict_train, data_dev[1][i])
    if clas in j:
        score += 1

