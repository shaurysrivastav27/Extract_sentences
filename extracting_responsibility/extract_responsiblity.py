import pandas as pd 
import numpy as np
import nltk
df= pd.read_csv("dataframe.csv",index_col='Unnamed: 0')

df.reset_index(drop=True,inplace=True)

df.drop('Unnamed: 0.1',axis=1,inplace=True)

## importing the neccessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
## skills dictionary
skilldict = {}
for job in df['Job Title'].unique():
    skills = []
    for i in df[df['Job Title']==job]['Skills'].map(lambda x: x.lower() if(type(x)==str) else x):
        if(type(i)==str):
            Skills = i.split()
        for j in nltk.pos_tag(Skills):
            if((j[1]=='NNP')):   ## extracting the skills which are NNP or proper nouns
                skills.append(j[0].lower())
    skilldict[job] = list(np.unique(skills))


## functions

df['new responsibility'] = ''

for i in range(0,len(df)):
    try:
        job = df['Job Title'][i]
        df['new responsibility'][i] = new_res(df['description'][i],job)
    except:
        df['new responsibility'] = 'None'

df['new responsibility'] = df['new responsibility'].map(lambda x:x.replace("Job Description"," ").lstrip() if(re.search("Job Description",x)) else x)

## chunking 
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from nltk import Tree

respon = ['responsible','responsibiltiy','responsibilities','function','activity','accountable','accountabilities']

def chunkI(i):
    if(i=='None'):
        return i
    rules = r"""Chunk:  {<DT>?<JJ>*<V.+>+<.*>*|<JJ.?>*<IN.?>*<VBG.?>|<NN.?>*<RB.?>*<IN.?>*|<NNS.?>*<CC.?>*<NNS.?>*<NN.?>*|<DT.?>*<NN.?>*<VBP.?>*|<VB.?>*<CC.?>?<VB.?>*<JJ.*>|<VB.?>*<JJ.?>*<NNS.?>*<VB.?>*<.*>|<RB.?>*<VB.?>*<NN.?>*<NN.?>*<.*>}
    """
    chunk = RegexpParser(rules)
    x = ""
    #for i in findf['new responsibilty']:
    for j in i.split("."):
        y = j+"."
        words = word_tokenize(y)
        flag = 1
        for ind in words:
            if(ind in respon):
                x = x + y;
                flag = 0
                break
        if(flag):
            postags = nltk.pos_tag(words)
            for child in chunk.parse(postags):
                if isinstance(child,Tree):  
                    if (child.label() == 'Chunk'):
                        x = x+y+" "
                        break
    return x
    
df['new responsibility'] = df['new responsibility'].map(chunkI)




