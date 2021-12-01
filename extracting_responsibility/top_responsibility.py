#calling the sentence transformer library
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name_or_path="bert-base-nli-mean-tokens") #using the bert model

import pandas as pd
import numpy as np

df = pd.read_csv('/home/shaury/Work/naukri.csv',index_col='Unnamed: 0')

#taking the samples
findf = df[df['Searched Job Title']=='Application Developer'].copy(deep=True)

#using punkt sentence tokenizer for tokenizing the sentences 

from nltk.tokenize import PunktSentenceTokenizer
punkt = PunktSentenceTokenizer()

#creating the sentence array
sentences = []
for i in range(0,len(findf)):
    if(findf['new responsibility'][i]!='None'):
        for j in punkt.tokenize(findf['new responsibility'][i]):
            sentences.append(j)
            

#defining the model
sentence_vecs = model.encode(sentences)

from sklearn.metrics.pairwise import cosine_similarity #word mover distance approach

## the final dictionary for the final data frame
mydict = {}
for i in sentences:
    mydict[i] = 0
    
## taking the sentences with significant similarity with other sentences
cnt = 0
for i in sentence_vecs:
    for j in cosine_similarity([i],sentence_vecs)[0]:
        if(j>0.65): #using 0.65 to get the most significant counts only
            mydict[sentences[cnt]] +=1
    cnt=cnt+1

pd.DataFrame([mydict]).transpose().sort_values(0,ascending=False)


