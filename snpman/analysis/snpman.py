#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os


# In[2]:


directory='/home/a/Desktop/snpman/mysnps'

for filex in os.scandir(directory):
    if ("23andme" in str(filex)) or ("example" in str(filex)):
        print(filex)
        with open(filex,'r') as oldfile, open(directory+(os.path.basename(filex))+"new.csv",'w') as newfile:
            print(filex)
            for line in oldfile:
                if "#" not in line:
                    newfile.write(line)
    


# In[3]:


directory='/home/a/Desktop/snpman'

for filex in os.scandir(directory):
    if ".txt" in os.path.basename(filex):
        gendf=pd.read_csv('../'+os.path.basename(filex), header=None, sep='\t')
        gendf.columns=['resid','chromosome','position','genotype']
        if "X" and "Y" in gendf.chromosome.unique():
            gender="M"
        else:
            gender="F"
        gendffilter=gendf[gendf.chromosome==1]
        gendffilter=gendffilter[gendffilter.genotype!="--"]
        with open('../example_genome.ped','w+') as pedfile:
            for row in gendffilter.resid:
                if row not in pedfile:
                    pedfile.write("M "+str(row)+"\n")


# In[6]:


directory='/home/a/Desktop/snpman'
for filex in os.scandir(directory):
    if (".csv" in str(filex)):
        gendf=pd.read_csv('../'+os.path.basename(filex), header=None, sep='\t')
        gendf.columns=['resid','chromosome','position','genotype']
        if "X" and "Y" in gendf.chromosome.unique():
            gender="M"
        else:
            gender="F"
        gendffilter=gendf[gendf.chromosome==1]
        gendffilter=gendffilter[gendffilter.genotype!="--"]
        with open('../haptypes.hap','a+') as hapfile:
            for row in gendffilter.resid:
                if row not in hapfile:
                    hapfile.write(str(row)+"\n")
        


# In[13]:


gendf=pd.read_csv('../mysnpsexample_genome.txtnew.csv', header=None, sep='\t')
gendf.columns=['resid','chromosome','position','genotype']
gendffilter=gendf[gendf.chromosome==1]
gendffilter.resid=gendffilter.resid.map(lambda x: x.lstrip("irs"))
gendffilter=gendffilter.sort_values(by='resid',ascending=False)
gendffilter.resid=pd.to_numeric(gendffilter.resid)
gendffilter=gendffilter[gendffilter.genotype!="--"]
#print(gendffilter.head())
#print(gendffilter.resid.count())
#print(gendffilter.iloc[0][0])

#print(gendffilter.head())
length=(gendffilter.max()[0])
#length=gendffilter.iloc[-1][0]
print(length)
'''
length=length.translate(str.maketrans('','','irs'))
print(length)
emptyarray=[" "]*(int(length)+1)
count=1
name="S_0"+str(count)
emptyarray[0]=name
emptyarray[1]=name
emptyarray[2]=str(0)
emptyarray[3]=gender
for id,geno in gendffilter.iterrows():
    #print(geno.resid,geno.genotype)
    new=""
    for letter in geno.resid:
        if not(letter.isalpha()):
            new+=letter
    print(int(new))
    emptyarray[int(new)]=geno.genotype
'''

