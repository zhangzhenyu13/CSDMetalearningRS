from DataPrepare.ConnectDB import *
from gensim import corpora
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics,preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import sparse
import json
import pickle
import copy
import gc
gc.collect()
warnings.filterwarnings("ignore")
##############################################################################
def onehotFeatures(data,threshold_num=5,threshhold_ration=0.01):
    '''

    :param data:str data
    :return: one-hot vector representation
    '''
    c = {}
    maxK=None
    for r in data:
        if r is None:
            continue
        xs = r.split(",")
        for x in xs:
            if x in c.keys():
                c[x] += 1
            else:
                c[x] = 1
            if maxK is None:
                maxK=x
            else:
                if c[x]>c[maxK]:
                    maxK=x
    rmKs=[]
    for k in c.keys():
        if c[k]<threshold_num or c[k]<c[maxK]*threshhold_ration:
            rmKs.append(k)
    for k in rmKs:
        del c[k]
    c=c.keys()
    i_c = {}
    count = 0
    for i in c:
        i_c[i] = count
        count += 1
    X = sparse.dok_matrix((len(data), count))
    row = 0
    for r in data:

        if r is None:
            row += 1
            continue

        xs = r.split(",")
        for x in xs:
            if x not in c:
                continue
            col = i_c[x]
            X[row, col] = 1
        row += 1
    #print("one-hot feature size=%d"%(len(c)),"removed feature size=%d"%(len(rmKs)))
    return (X.toarray(),len(c))

def showData(X):
    import Utility.personalizedSort as ps
    m_s=ps.MySort(X)
    m_s.compare_vec_index=-1
    y=m_s.mergeSort()
    x=np.arange(len(y))
    plt.plot(x,np.array(y)[:,0])
    plt.show()

class Vectorizer:
    def loadData(self):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid,detail,taskname, duration,technology,languages,prize,postingdate,diffdeg,tasktype from task order by postingdate desc'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.ids=[]
        self.docs=[]
        self.duration=[]
        self.prize=[]
        self.techs=[]
        self.lan=[]
        self.startdate=[]
        self.diffdeg=[]
        self.tasktype=[]
        for data in dataset:
            #print(data)
            self.ids.append(data[0])

            if data[1] is None:
                self.docs.append(data[2])
            else:
                self.docs.append(data[2]+"\n"+data[1])
            if data[3]>50:
                self.duration.append([50])
            elif data[3]<1:
                self.duration.append([1])
            else:
                self.duration.append([data[3]])
            self.techs.append(data[4])
            self.lan.append(data[5])

            if data[6]!='':
                prize=np.sum(eval(data[6]))
                if prize>6000:
                    prize=6000
                if prize<1:
                    prize=1
                self.prize.append([prize])
            else:
                self.prize.append([1.])

            if data[7]<1:
                self.startdate.append([1])
            else:
                self.startdate.append([data[7]])

            if data[8]>0.6:
                self.diffdeg.append([0.6])
            else:
                self.diffdeg.append([data[8]])

            self.tasktype.append(data[9])
        #print(self.ids[:6])
        print("task num=%d" % len(self.ids))
        #showData(self.diffdeg)
        #showData(self.prize)
        #showData(np.log(self.prize).tolist())
        #showData(self.duration)
        #showData(np.log(self.duration).tolist())
        #showData(self.startdate)
        #showData(np.log(self.startdate).tolist())

class LDAFlow(Vectorizer):
    def __init__(self):
        self.n_features=200

    def cleanDocs(self,docs_o):
        docs=copy.deepcopy(docs_o)
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        lemma = WordNetLemmatizer()
        for i in range(len(docs)):
            doc = docs[i]
            doc = " ".join([i for i in doc.lower().split() if i not in stop])
            doc = ''.join(ch for ch in doc if ch not in exclude)
            doc = " ".join(lemma.lemmatize(word) for word in doc.split())
            docs[i] = doc.split()
        return docs

    def transformVec(self,docs):
        #print(np.shape(docs),docs[0])
        docs=self.cleanDocs(docs)

        X = sparse.dok_matrix((len(docs), self.n_features))
        row = 0
        for doc in docs:
            doc_bow = self.dictionary.doc2bow(doc)
            lda_doc = self.lda[doc_bow]
            # print(type(lda_doc),lda_doc)
            for topic in lda_doc:
                X[row, topic[0]] = topic[1]
            row += 1
        return X.toarray()

    def train_doctopics(self,docs):
        #print(np.shape(docs),docs[0])
        t0 = time.time()

        docs = self.cleanDocs(docs)

        print("performing LDA ")
        self.dictionary = corpora.Dictionary(docs)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in docs]
        self.lda = gensim.models.LdaModel(doc_term_matrix, num_topics=self.n_features, id2word=self.dictionary)
        print("LDA built in %fs" % (time.time() - t0))
        with open("../data/ldamodel.pkl","wb") as f:
            model={}
            model["n_features"]=self.n_features
            model["dict"]=self.dictionary
            model["lda"]=self.lda
            pickle.dump(model,f)
    def loadModel(self):
        print("loading lda model")
        with open("../data/ldamodel.pkl","rb") as f:
            model=pickle.load(f)
            self.n_features=model["n_features"]
            self.dictionary=model["dict"]
            self.lda=model["lda"]

#######################################################################################################################
def hashingIDF(n_features):
    # Perform an IDF normalization on the output of HashingVectorizer
    hasher = HashingVectorizer(n_features=n_features,
                                stop_words='english', alternate_sign=False,
                                norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())

    return vectorizer

def IDF(n_features):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    return vectorizer

class LSAFlow(Vectorizer):
    def __init__(self):
        self.n_features=200

    def transformVec(self,docs):
        X=IDF(self.n_features*10).fit_transform(docs)
        X = self.lsa.fit_transform(X)
        return X
    def train_doctopics(self,docs):
        t0 = time.time()

        X = IDF(self.n_features*10).fit_transform(docs)
        print("Performing  LSA")
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(self.n_features)
        normalizer = Normalizer(copy=False)
        self.lsa = make_pipeline(svd, normalizer)
        self.lsa.fit_transform(X)
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print("LSA built in %fs" % (time.time() - t0))
        with open("../data/lsamodel.pkl","wb") as f:
            model={}
            model["n_features"]=self.n_features
            model["lsa"]=self.lsa
            pickle.dump(model,f)

    def loadModel(self):
        print("loading lsa model")
        with open("../data/lsamodel.pkl","rb") as f:
            model=pickle.load(f)
            self.n_features=model["n_features"]
            self.lsa=model["lsa"]

#######################################################################################################################

# Do the actual clustering
n_clusters=20

def KM_cluster(X,true_k,minibatch=False):
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=True)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=True)

    print("Clustering sparse data with %s" % km)
    km.fit(X)
    print()
    return km

#evalute cluster result in several metrics
def evaluateCluster(X,labels,km):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    print()

def scaler(X):
    minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax.fit_transform(X)
    return minmax.transform(X)

def taskVec(model,X):
    X_techs,features1 = onehotFeatures(model.techs)
    X_lans,features2 = onehotFeatures(model.lan)
    X_tasktype,features3 = onehotFeatures(model.tasktype)

    X_startdate = np.log(model.startdate)
    X_duration = np.log(model.duration)
    X_prize = np.log(model.prize)
    X_diffdeg = model.diffdeg

    features_num=(model.n_features,features1,features2,features3)

    X = np.concatenate((X,X_techs), axis=1)
    X = np.concatenate((X, X_lans), axis=1)
    X = np.concatenate((X, X_tasktype), axis=1)
    X = np.concatenate((X, X_startdate), axis=1)
    X = np.concatenate((X, X_duration), axis=1)
    X = np.concatenate((X, X_prize), axis=1)
    X = np.concatenate((X, X_diffdeg), axis=1)
    return (X,features_num)
def weightedTaskVec(model,X):
    #weight of topics,techs,languages,postingdate,duration,prize,diffdeg,tasktype
    w=[1.0,2.0,1.5,4.0,3.0,1.0,4.0,2.0]
    X_techs=scaler(onehotFeatures(model.techs))
    X_lans=scaler(onehotFeatures(model.lan))
    X_tasktype=scaler(onehotFeatures(model.tasktype))

    X_startdate=scaler(np.log(model.startdate))
    X_duration=scaler(model.duration)
    X_prize=scaler(model.prize)
    X_diffdeg=scaler(model.diffdeg)
    #print(X_techs[:3])
    X=np.concatenate((w[0]*X,w[1]*X_techs),axis=1)
    X=np.concatenate((X,w[2]*X_lans),axis=1)
    X=np.concatenate((X,w[3]*X_tasktype),axis=1)
    X = np.concatenate((X, w[4] * X_startdate), axis=1)
    X = np.concatenate((X, w[5] * X_duration), axis=1)
    X = np.concatenate((X, w[6] * X_prize), axis=1)
    X = np.concatenate((X, w[7] * X_diffdeg), axis=1)
    return X

#save data and load data
def saveTaskVecData(X,taskid,feature_num,choice=1):
    data={}
    data["size"]=len(taskid)
    data["feature_num"]=feature_num
    data["taskids"]=taskid
    data["tasks"]=X
    with open("../data/clusterResult/taskVec"+str(choice)+".data","wb") as f:
        pickle.dump(data,f)


def loadCluster(Train=False,splitratio=0.8,choice=1):
    with open("../data/clusterResult/taskVec"+str(choice)+".data","rb") as f:
        data=pickle.load(f)
    size=data["size"]
    taskids=data["taskids"]
    X=data["tasks"]
    trainSize=int(size*splitratio)

    if Train:
        X=X[:trainSize]
        taskids=taskids[:trainSize]
    else:
        X=X[trainSize:]
        taskids=taskids[trainSize:]

    print("loaded data size=%d"%len(taskids))
    return np.array(taskids),np.array(X)

def genResults():

    choice=eval(input("1:LDA; 2:LSA \t"))
    if choice==1:
        lda=LDAFlow()

        model=lda
    else:
        choice=2
        lsa=LSAFlow()

        model=lsa
    #load model
    model.loadData()
    model.loadModel()
    #model.train_doctopics(model.docs)
    X=model.transformVec(model.docs)
    X,feature_num=taskVec(model,X)
    taskid=model.ids
    #save vec representaion of all the tasks
    saveTaskVecData(X,taskid,feature_num,choice)
    taskid,X=loadCluster(Train=True,choice=choice)

    print("training for clustering, tasks size=%d"%len(X))
    n_clusters=20
    taskClusters=None

    while n_clusters>0:
        km=KM_cluster(X,n_clusters,minibatch=False)
        print("n_samples: %d, n_features: %d" % X.shape)
        print()
        with open("../data/clusterResult/kmeans" + str(choice) + ".pkl", "wb") as f:
            pickle.dump(km, f)
        with open("../data/clusterResult/kmeans" + str(choice) + ".pkl", "rb") as f:
            km=pickle.load(f)

        result=km.predict(X)
        taskClusters={}
        for i in range(n_clusters):
            taskClusters[i]=[]
        for i in range(len(result)):
            c_no=result[i]
            taskClusters[c_no].append(taskid[i])

        #plot result
        hist=[(k,len(taskClusters[k])) for k in taskClusters.keys()]
        for i in hist:
            print(i)
        plt.plot(hist,marker='o')
        plt.title("choice=%d"%choice)
        plt.show()
        n_clusters=eval(input("current cluster size is %d    "%n_clusters))

    #saving result
    print("saving clustering result")
    with open("../data/clusterResult/clusters" + str(choice) + ".data", "wb") as f:
        pickle.dump(taskClusters,f)



if __name__ == '__main__':
    genResults()
