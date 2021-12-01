import pandas as pd
import numpy as np
import nltk

df = pd.read_csv("dataframe_containing_job_descriptions.csv",index_col='Unnamed: 0')
df.head()

jobs = pd.read_excel("/Work/files/Semantic Titles.xlsx",sheet_name='Sheet1')
jobs.head(2)

## MAIN FUNCTION 

def skillarray(X,job):
    x = X[X['Searched Job Title']==job]
    if(len(x)==0):
        return np.array(['None','None','None','None','None'])
    alllist = []
    findict = {}
    for i in x['skills']:
        for j in i.split():
            try:
                findict[j.lower()]+=1
            except:
                findict[j.lower()] = 0
            alllist.append(j.lower())
    alllist = list(np.unique(np.array(alllist)))
    skillset = pd.DataFrame([findict]).transpose()
    skillset.reset_index(inplace=True)
    skillset.columns = ['skills','counts']
    skillset.sort_values('counts',ascending=False,inplace=True)
    skillset['percentage'] = (skillset['counts']/len(x))*100
    skillset['percentage'] = skillset['percentage'].astype(int)
    skillset['fin'] = skillset['skills']+" ("+skillset['percentage'].astype(str)+"%)"
    return np.array(skillset['fin'][0:5])
    

## MAKING THE DATAFRAME 

skilldict = {}
for i in jobs['Job Title']:
    skilldict[i] = skillarray(df,i)

## final dataframe

pd.DataFrame(skilldict).transpose()


