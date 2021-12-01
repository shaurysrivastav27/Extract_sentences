def sentence_maker(sentence):
	'''
		Input : A string of whole description text.
		Return values : 
			sentences : list of sentences in the for of tokenized words.
			full_Sent : list of sentences in the for of tokenized sentences from the main text.  
	'''
	sentences = []
	full_sent = []
	for i in punkt.tokenize(sentence):
		if(len(i.split())>3):
                sentences.append(i.lower().split())
                full_sent.append(i)
	return sentences,full_sent
        
        
W = Word2Vec(sentences,min_count = 1)


def vectorizer(sent,W):
    vec = []
    numw = 0
    for i in sent:
        try:
            if(numw==0):
                vec = W[i]
            else:
                vec = np.add(vec,W[i])
            numw+=1
        except:
            pass
    return np.asarray(vec)/numw
l = []
for sent in sentences:
    l.append(vectorizer(sent,W))
X = np.array(l)

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
arr = []
silhouette_avg = []
for i in range(2,10):
    kmeans = KMeans(n_clusters = i , init = "k-means++")
    kmeans.fit(X)
    arr.append(kmeans.inertia_)
    silhouette_avg.append(silhouette_score(X, kmeans.labels_))


## optimal k = 5
model = KMeans(n_clusters=5, init= 'k-means++')
model.fit(X)
sentdf = pd.DataFrame({"Sent":full_sent,"label":model.predict(X)})
pd.DataFrame({"Sent":full_sent,"label":model.predict(X)})

### similarity        
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name_or_path="bert-base-nli-mean-tokens") #using the bert model
top_resp = []
for lab in range(0,5):
    sentences = []
    for sent in sentdf[sentdf['label']==lab]['Sent']:
        sentences.append(sent)
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

    dftemp = pd.DataFrame([mydict]).transpose().sort_values(0,ascending=False).reset_index()
    dftemp.columns = ['Sent','count']
    top_resp.append(dftemp['Sent'][0]+". "+dftemp['Sent'][1]+". "+dftemp['Sent'][2])











