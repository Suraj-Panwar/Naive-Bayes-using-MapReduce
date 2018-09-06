
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import nltk
import pickle
import gensim
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import time


# In[42]:

# Read Data
df_train = pd.read_table('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\train.csv' , sep='\t<', header=None)
print('Train Done')
df_dev = pd.read_table('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\dev.csv' , sep='\t<', header=None)
print('Dev Done')
df_test = pd.read_table('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\test.csv' , sep='\t<', header=None)
print('Test Done')


# In[43]:


# Only Run Once
df_train[0] = [x[1:] for x in df_train[0]]
df_dev[0] = [x[1:] for x in df_dev[0]]
df_test[0] = [x[1:] for x in df_test[0]]


# In[44]:


# Only Run Once
df_train[0] = [x.split(',') for x in df_train[0]]
df_dev[0] = [x.split(',') for x in df_dev[0]]
df_test[0] = [x.split(',') for x in df_test[0]]


# In[45]:

# for timing the data training
start = time.time()


# In[46]:


# Make Dictionary
dict_train = {}
for item, value in zip(df_train[0], df_train[1]):
    for small_item in item:
        small_item = small_item.strip(' ')
        if small_item in dict_train:
            dict_train[small_item] = dict_train[small_item] + [value]
        else:
            dict_train[small_item] = [value]


# In[47]:


for item in dict_train.keys():
    dict_train[item] =  ' '.join(dict_train[item])


# In[48]:


dict_train_final = {}
for key in dict_train.keys():
    #print(key)
    dict_train_final[key] = nltk.FreqDist([word for word in gensim.utils.simple_preprocess(dict_train[key]) if word not in stopwords][11:])


# In[49]:


end = time.time()
print('Time Takem:', end - start)


# In[50]:


# For Storing Dictionary
# with open('dictionary.pickle', 'wb') as handle:
#     pickle.dump(dict_train_final, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[51]:


lis = []
for key in dict_train:
    lis.append(dict_train[key])
final_string = ' '.join(lis)
main_dict = nltk.FreqDist([word for word in gensim.utils.simple_preprocess(final_string) if word not in stopwords])


# In[52]:


# For Storing Dictionary
# with open('main_dictionary.pickle', 'wb') as handle:
#     pickle.dump(main_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[53]:

# main  function computing the accuracy
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
    return ret


# In[54]:

# Provides the accuracy
score = 0
total = 0
for i, j in zip(range(len(df_dev)), df_dev[0]):
    total +=1 
    if total%1000 ==0:
        print('Accuracy: {0:1.2f}, Samples: {1:5.0f}'.format((score/total)*100, total))
    clas = find_class(main_dict, dict_train_final, df_dev[1][i])
    if clas in j:
        score += 1

