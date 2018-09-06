
# coding: utf-8

# In[3]:

# For data correction and simple preprocessing 
import numpy as np
import pandas as pd
import nltk


# In[74]:


df = pd.read_fwf("C:/Users//suraj//Desktop//MLLLD//Assignment-1/full_train.txt")


# In[76]:


df1 = df.loc[df['Unnamed: 1'].isnull(),:]


# In[78]:


df1.to_csv("C:\\Users\\suraj\\Desktop\\MLLLD\\Assignment-1\\train.csv", header=False, index=False)

