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
from ML_Models.Model_def import ML_model
import pickle
import copy
import gc
gc.collect()
warnings.filterwarnings("ignore")
##############################################################################
def onehotFeatures(data,threshold_num=5):
    '''
    :param data:str data
    :return: one-hot vector representation
    '''
    c = {}
    for r in data:
        if r is None:
            continue
        xs = r.split(",")
        for x in xs:
            if x in c.keys():
                c[x] += 1
            else:
                c[x] = 1

    rmKs=[]
    for k in c.keys():
        if c[k]<threshold_num :
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
    def filterData(self,threshold=100):
        tasktype_dict={}
        for i in range(len(self.tasktype)):
            t=self.tasktype[i]
            if t is None:
                continue
            if t in tasktype_dict.keys():
                tasktype_dict[t].append(self.ids[i])
            else:
                tasktype_dict[t]=[]
                tasktype_dict[t].append(self.ids[i])
        rmType=set()
        rmsize=0
        keepType={}
        for k in tasktype_dict.keys():
            if len(tasktype_dict[k])<threshold:
                rmType.add(k)
                rmsize += len(tasktype_dict[k])
            else:
                keepType[k]=tasktype_dict[k]
        with open("../data/clusterResult/tasktypeCluster.data","wb") as f:
            pickle.dump(keepType,f)
        with open("../data/clusterResult/tasktypeCluster.data", "rb") as f:
            keepType=None
            keepType=pickle.load(f)
            #for k in keepType.keys():print(k,len(keepType[k]))
        rmIndices=[]
        for i in range(len(self.tasktype)):
            if self.tasktype[i] in rmType:
                rmIndices.append(i)
        print("num to remove",len(rmIndices),rmsize)
        n=len(self.tasktype)
        for i in range(1,(n+1)):
            if len(rmIndices)==0:
                break
            index=n-i

            if index ==rmIndices[len(rmIndices)-1]:
                #print("type size",tasktype_dict[self.tasktype[index]])
                rmIndices.pop()
                self.ids.pop()
                self.tasktype.pop()
                self.docs.pop()
                self.duration.pop()
                self.prize.pop()
                self.techs.pop()
                self.lan.pop()
                self.startdate.pop()
                self.diffdeg.pop()
        print("after filter,size=",len(self.ids))

    def loadData(self):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid,detail,taskname, duration,technology,languages,prize,postingdate,diffdeg,tasktype from task ' \
                 ' order by postingdate desc'
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
            if data[1] is None:
                continue
            self.ids.append(data[0])
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
        self.n_features=1000

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
        self.n_features=1000

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

class ClusteringModel(ML_model):
    def __init__(self):
        ML_model.__init__(self)
        self.model={}

        with open("../data/clusterResult/tasktypeCluster.data", "rb") as f:
            self.dataSet = pickle.load(f)
            print("%d original types" % len(self.dataSet.keys()))
        taskstypes=self.dataSet.keys()
        for t in taskstypes:
            self.model[t]=None
            self.dataSet[t]=np.array(self.dataSet[t])

    def trainCluster(self,X,tasktype,n_clusters=1,minibatch=False):

        if minibatch:
            km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=True)
        else:
            km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                        verbose=True)

        print("Clustering sparse data with %s" % km)
        km.fit(X)
        print()
        self.model[tasktype]=km
    def predictCluster(self,X,tasktype):
        return  self.model[tasktype].predict(X)


def clusterVec(model,docX):
    #weight of topics,techs,languages,postingdate,duration,prize,diffdeg
    X_techs,_=onehotFeatures(model.techs)
    X_lans,_=onehotFeatures(model.lan)

    print(X_techs.shape,docX.shape)
    X=np.concatenate((docX,X_techs),axis=1)
    X=np.concatenate((X,X_lans),axis=1)

    return X

def taskVec(model,docX):
    X_techs,features1 = onehotFeatures(model.techs)
    X_lans,features2 = onehotFeatures(model.lan)

    print("techs/lans",features1,features2)

    X_startdate = model.startdate
    X_duration = model.duration
    X_prize = model.prize
    X_diffdeg =model.diffdeg

    features_num={"doc_topics":model.n_features,"techs":features1,"languages":features2}

    X = np.concatenate((docX,X_techs), axis=1)
    X = np.concatenate((X, X_lans), axis=1)
    X = np.concatenate((X, X_startdate), axis=1)
    X = np.concatenate((X, X_duration), axis=1)
    X = np.concatenate((X, X_prize), axis=1)
    X = np.concatenate((X, X_diffdeg), axis=1)
    return (X,features_num)

#save data and load data
def saveTaskVecData(X,taskid,feature_num,choice=1):
    data={}
    data["size"]=len(taskid)
    data["feature_num"]=feature_num
    data["taskids"]=taskid
    data["tasks"]=X
    with open("../data/clusterResult/taskVec"+str(choice)+".data","wb") as f:
        pickle.dump(data,f)

def loadTaskVecData(Train=False,splitratio=0.8,choice=1):
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
    model.filterData(100)
    model.loadModel()
    #model.train_doctopics(model.docs)
    docX=model.transformVec(model.docs)
    X,feature_num=taskVec(model,docX)
    taskid=model.ids
    #save vec representaion of all the tasks
    saveTaskVecData(X,taskid,feature_num,choice)
    #cluster task based on its feature
    X=clusterVec(model,docX)

    print("training for clustering, tasks size=%d"%len(X))
    clusterEXP=1000
    IDClusters={}
    model=ClusteringModel()
    model.name="clusteringModel"+str(choice)
    for k in model.dataSet.keys():
        n_clusters=max(1,len(model.dataSet[k])//clusterEXP)
        localids=model.dataSet[k]
        localX=[]
        localIDs=[]
        for i in range(len(taskid)):
            id=taskid[i]
            if id in localids:
                localX.append(X[i])
                localIDs.append(id)
        localX=np.array(localX)
        model.trainCluster(X=localX,tasktype=k,n_clusters=n_clusters,minibatch=False)

        result=model.predictCluster(localX,k)
        for j in range(len(result)):
            r=result[j]
            t=k+str(r)
            if t not in IDClusters.keys():
                IDClusters[t]=[localIDs[j]]
            else:
                IDClusters[t].append(localIDs[j])

    model.saveModel()
    model.loadModel()
    # saving result
    print("saving clustering result")
    with open("../data/clusterResult/clusters" + str(choice) + ".data", "wb") as f:
        pickle.dump(IDClusters, f)

    with open("../data/clusterResult/clusters" + str(choice) + ".data", "rb") as f:
        taskidClusters=pickle.load(f)

    #plot result
    hist = [(k,len(taskidClusters[k])) for k in taskidClusters.keys()]
    for i in hist:
        print(i)
    plt.plot(np.arange(len(hist)),hist, marker='o')
    plt.title("choice=%d" % choice)
    plt.show()


if __name__ == '__main__':
    genResults()

