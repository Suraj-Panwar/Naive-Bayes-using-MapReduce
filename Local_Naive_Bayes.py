
# coding: utf-8

# In[1]:


###### Import Modules ######

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import time


# In[2]:


###### We use NLTK library for lemmatization and stopwords ######

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))


# In[3]:


####### Read Data for making a Dictionary ######

with open('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\full_train.txt') as file:
    content = file.readlines()


# In[4]:


###### Begin Clock ######

start_time = time.time()


# In[14]:


'''
This Function carries out simple functions that are used for preprocessing the data i.e striping end whitespaces, lowering the 
string and carrying out lemmatization. This is used in all modules

'''

def preprocess(line):
    labels, sentence = line.split('\t',1) ### Split text and classes ###
    sentence = sentence.lower()
    
    sentence =  ' '.join([x for x in sentence.split() if ('\\' not in x) and ('<' not in x)])
    sentence = ''.join([x for x in sentence if (not x.isdigit() and (x not in string.punctuation)) ])
    
    text = sentence.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
    return labels, sentence, text


# In[15]:


###### Construct Empty Dictionaries ######

dict_train = {}
dict_label = {}
voc = set()

####### Construct complete Dictionaries #######

for line in content:
    
    labels, sentence, text = preprocess(line)
    
    for label in labels.split(','):
        label = label.strip()
        if label not in dict_label:
            dict_label[label] = 1
        else:
            dict_label[label] += 1
            
        if label not in dict_train:
            dict_train[label] = {}
            
        for word in text:
            if len(word)>2:
                if word not in dict_train[label]:
                    dict_train[label][word] = 1
                else:
                    dict_train[label][word] += 1 
            voc.add(word)


# In[16]:


###### Calculate Prior Probabilities for calculating the final class estimation ######

prior= {}
total_label = sum(dict_label.values())

for key in dict_label.keys():
    prior[key] = np.log(dict_label[key] / total_label)
    
vocab_size  = len(voc)
vocab = list(voc)


# In[17]:


###### For calculating total word count ######

for label in dict_train.keys():
    
    total_words = sum(dict_train[label].values())
    dict_train[label]['words_in_class'] = total_words


# In[18]:


###### For calculating the total time taken for training ######

end_time = time.time()
print('Time Takem for training is :', end_time - start_time)


# In[26]:


'''
This is the main program that takes the trained libraries from the previous excercises and predicts the final class of a given
sample, all the samples are serially passed through the algorithm and the correct classification is considered if the predicted 
class is one of the true class. Data set is the train dataset.

'''

correct = 0 
m = 1
v = vocab_size

count = 0

for line in content:
    count +=1
    if count%10000 ==0:
        print('{} Samples Processed so far.'.format(count))
    
    
    labels, sentence, text = preprocess(line)
    true_labels = [label.strip() for label in labels.split(',')]
    prob_den = {}
    
    for class_label in dict_train.keys():
        total_class =  dict_train[class_label]['words_in_class']
        
        for word in text:
            
            if (word in dict_train[class_label]) and len(word):
                count_fo_w_in_class = dict_train[class_label][word]
                
            else:
                count_fo_w_in_class = 0
                
            try:
                prob_den[class_label]=prob_den[class_label]+np.log((count_fo_w_in_class + m/v)/(total_class+1))
                
            except:
                prob_den[class_label]=np.log((count_fo_w_in_class + m/v)/(total_class+1))
        prob_den[class_label] = prob_den[class_label] + prior[class_label]
        
    max_prob = -9999999
    
    for class_label in prob_den.keys():
        
        if(prob_den[class_label] >= max_prob):
            pred_class = class_label
            max_prob = prob_den[class_label]
            
    if(pred_class in true_labels):
        correct = correct + 1

accuracy = correct / len(content) * 100
print("Train Accuracy : {}, Total Samples Classified : {} ".format(accuracy, count))


# In[27]:


###### Read Development Dataset ######

with open('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\full_devel.txt') as file:
    content_devel = file.readlines()


# In[28]:


'''
This is the main program that takes the trained libraries from the previous excercises and predicts the final class of a given
sample, all the samples are serially passed through the algorithm and the correct classification is considered if the predicted 
class is one of the true class. Data set is the Development dataset.

'''
correct = 0 
m = 1
v = vocab_size

count = 0

for line in content_devel:
    count +=1
    if count%10000 ==0:
        print('{} Samples Processed so far.'.format(count))
    
    
    labels, sentence, text = preprocess(line)
    true_labels = [label.strip() for label in labels.split(',')]
    prob_den = {}
    
    for class_label in dict_train.keys():
        total_class =  dict_train[class_label]['words_in_class']
        
        for word in text:
            
            if (word in dict_train[class_label]) and len(word):
                count_fo_w_in_class = dict_train[class_label][word]
                
            else:
                count_fo_w_in_class = 0
                
            try:
                prob_den[class_label]=prob_den[class_label]+np.log((count_fo_w_in_class + m/v)/(total_class+1))
                
            except:
                prob_den[class_label]=np.log((count_fo_w_in_class + m/v)/(total_class+1))
        prob_den[class_label] = prob_den[class_label] + prior[class_label]
        
    max_prob = -9999999
    
    for class_label in prob_den.keys():
        
        if(prob_den[class_label] >= max_prob):
            pred_class = class_label
            max_prob = prob_den[class_label]
            
    if(pred_class in true_labels):
        correct = correct + 1

accuracy = correct / len(content_devel) * 100
print("Development Accuracy : {}, Total Samples Classified : {} ".format(accuracy, count))


# In[31]:


###### Read test Dataset ######


with open('C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\full_test.txt') as file:
    content_test = file.readlines()


# In[32]:


'''
This is the main program that takes the trained libraries from the previous excercises and predicts the final class of a given
sample, all the samples are serially passed through the algorithm and the correct classification is considered if the predicted 
class is one of the true class. Data set is the test dataset.

'''

correct = 0 
m = 1
v = vocab_size

count = 0

for line in content_test:
    count +=1
    if count%10000 ==0:
        print('{} Samples Processed so far.'.format(count))
    
    
    labels, sentence, text = preprocess(line)
    true_labels = [label.strip() for label in labels.split(',')]
    prob_den = {}
    
    for class_label in dict_train.keys():
        total_class =  dict_train[class_label]['words_in_class']
        
        for word in text:
            
            if (word in dict_train[class_label]) and len(word):
                count_fo_w_in_class = dict_train[class_label][word]
                
            else:
                count_fo_w_in_class = 0
                
            try:
                prob_den[class_label]=prob_den[class_label]+np.log((count_fo_w_in_class + m/v)/(total_class+1))
                
            except:
                prob_den[class_label]=np.log((count_fo_w_in_class + m/v)/(total_class+1))
        prob_den[class_label] = prob_den[class_label] + prior[class_label]
        
    max_prob = -9999999
    
    for class_label in prob_den.keys():
        
        if(prob_den[class_label] >= max_prob):
            pred_class = class_label
            max_prob = prob_den[class_label]
            
    if(pred_class in true_labels):
        correct = correct + 1

accuracy = correct / len(content_test) * 100
print("Test Accuracy : {}, Total Samples Classified : {} ".format(accuracy, count))

