import pickle
import numpy as np
from Utility.personalizedSort import MySort
class ML_model:
    def __init__(self):
        self.model=None
        self.name=""

    def predict(self,X):
        '''
        predict the result based on given X
        :param X: input samples,(n,D)
        :return: given result, class or a real num
        '''
    def trainModel(self,dataSet):
        pass

    def findPath(self):
        modelpath="../data/saved_ML_models/classifiers/"+self.name+".pkl"
        return modelpath
    def loadModel(self):
        with open(self.findPath(),"rb") as f:
            data=pickle.load(f)
            self.model=data["model"]
            self.name=data["name"]
    def saveModel(self):
        with open(self.findPath(),"wb") as f:
            data={}
            data["model"]=self.model
            data["name"]=self.name
            pickle.dump(data,f)

#acc metrics
def getSubnumOfDIG(tasktype):
    with open("../data/UserInstances/UserGraph/SubNumBased/"+tasktype+"-UserInteraction.data","rb") as f:
        dataRank=pickle.load(f)
    return dataRank

def getScoreOfDIG(tasktype):
    with open("../data/UserInstances/UserGraph/ScoreBased/"+tasktype+"-UserInteraction.data","wb") as f:
        rankData=pickle.load(f)
    return rankData

def reRankSubUsers(rankData,taskid,topN=20):
    userRank=rankData[taskid]["ranks"]
    return np.array(userRank[:topN,0],dtype=np.int)

def reRankWinUsers(rankData,taskid,topN=20):
    userRank=rankData[taskid]["ranks"]
    return np.array(userRank[:topN,0],dtype=np.int)

def topKAccuracyWithDIG(Y_predict2,data,k,adding=False):
    '''
    :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                      else return false
    :param Y_predict2: a list of recommended entries
    :param data: the data set containing actual labels
    :return: boolean
    '''
    print("using DIG,adding=",adding)
    # measure top k accuracy
    dataRank=getSubnumOfDIG(data.tasktype)

    # batch data into task centered array
    ids=data.taskids[:data.testPoint]
    indexData=data.indexDataPoint(ids)
    #print(indexData);exit(12)
    left=indexData[0][1]

    Y_predict=[]
    Y_true=[]

    for step in range(1,len(indexData)):
        right=indexData[step][1]
        taskid=indexData[step-1][0]
        userRank=dataRank[taskid]["ranks"]

        trueY=data.testLabel[left:right]
        predictY=Y_predict2[left:right]
        #print(len(trueY),len(predictY))
        predictY=np.where(predictY==1)[0]
        predictY=set(predictY)

        if len(predictY)<k:
            #add users to meet requirement
            pos=0
            while len(predictY)<k:
                predictY.add(userRank[pos][0])
                pos+=1
        elif len(predictY)>=k and adding:
            #resort results using DIG
            t_predictY=[]
            for i in  range(len(userRank)):
                ranks=int(userRank[i][0])
                if ranks in predictY:
                    t_predictY.append(ranks)

            predictY=t_predictY

        Y_predict.append(np.array(list(predictY),dtype=np.int))
        Y_true.append(np.where(trueY==1)[0])

        left=right

    right=len(ids)
    taskid=ids[right-1]
    userRank=dataRank[taskid]["ranks"]

    trueY=data.testLabel[left:right]
    predictY=Y_predict2[left:right]
    predictY=np.where(predictY==1)[0]
    predictY=set(predictY)

    if len(predictY)<k:
        #add users to meet requirement
        pos=0
        while len(predictY)<k:
            predictY.add(userRank[pos][0])
            pos+=1

    elif len(predictY)>=k and adding:
            #resort results using DIG
        t_predictY=[]
        for i in  range(len(userRank)):
            ranks=int(userRank[i][0])
            if ranks in predictY:
                t_predictY.append(ranks)

        predictY=t_predictY

    Y_predict.append(np.array(list(predictY),dtype=np.int))
    Y_true.append(np.where(trueY==1)[0])

    Y=np.zeros(shape=len(Y_true))

    for i in range(len(Y_true)):
        #print(Y_true[i],Y_predict[i])
        #print(set(Y_true[i]).intersection(Y_predict[i]))
        tag=0

        for ele in Y_predict[i][:k]:
            if len(Y_true[i])==0:
                continue
            if ele in Y_true[i]:
                tag=1
                break

        Y[i]=tag

    return np.array(Y)


def topKAccuracy(Y_predict2,data,k):
    # measure top k accuracy
    # batch data into task centered array
    print("without DIG")

    ids=data.taskids[:data.testPoint]
    indexData=data.indexDataPoint(ids)
    #print(indexData);exit(12)
    left=indexData[0][1]

    Y_predict=[]
    Y_true=[]

    for step in range(1,len(indexData)):
        right=indexData[step][1]

        trueY=data.testLabel[left:right]
        predictY=Y_predict2[left:right]
        predictY=np.where(predictY==1)[0]

        Y_predict.append(np.array(list(predictY),dtype=np.int))
        Y_true.append(np.where(trueY==1)[0])

        left=right

    right=len(ids)

    trueY=data.testLabel[left:right]
    predictY=Y_predict2[left:right]
    predictY=np.where(predictY==1)[0]

    Y_predict.append(np.array(list(predictY),dtype=np.int))
    Y_true.append(np.where(trueY==1)[0])

    Y=np.zeros(shape=len(Y_true))

    for i in range(len(Y_true)):
        #print(Y_true[i],Y_predict[i])
        #print(set(Y_true[i]).intersection(Y_predict[i]))

        tag=False

        for ele in Y_predict[i][:k]:
            if len(Y_true[i])==0:
                continue
            if ele in Y_true[i]:
                tag=True
                break

        Y[i]=tag

    return np.array(Y)
