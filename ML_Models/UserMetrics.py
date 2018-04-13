import numpy as np
from Utility.TagsDef import getUsers
import pickle
from Utility.personalizedSort import MySort
#metrics

#subnum based rank data
def getSubnumOfDIG(tasktype):
    with open("../data/UserInstances/UserGraph/SubNumBased/"+tasktype+"-UserInteraction.data","rb") as f:
        dataRank=pickle.load(f)
    return dataRank

#score based rank data
def getScoreOfDIG(tasktype):
    with open("../data/UserInstances/UserGraph/ScoreBased/"+tasktype+"-UserInteraction.data","rb") as f:
        rankData=pickle.load(f)
    return rankData

#top n users after page rank
def getTopNUsersOnDIG(rankData,taskid,topN=20):
    userRank=rankData[taskid]["ranks"]
    return np.array(userRank[:topN,0],dtype=np.int)

#clip indices of top k data from given vector X
def getTopK(X,k):
    x_vec=[]
    for i in range(len(X)):
        x_vec.insert(i,[i,X[i]])

    mysort=MySort(x_vec)
    mysort.compare_vec_index=-1
    x_vec=mysort.mergeSort()
    x_vec=np.array(x_vec)[:k]

    return np.array(x_vec[:,0],dtype=np.int)

#select top k users based on its prediction possibility
def topKPossibleUsers(Y_predict,data,k):

    usersList=getUsers(data.tasktype)
    Y_label=data.testLabel

    taskNum=len(Y_label)//len(usersList)
    Y=np.zeros(shape=taskNum,dtype=np.int)
    print("top %d users from possibility for %d tasks(%d winners,%d users) "%
          (k,taskNum,np.sum(Y_label),len(usersList)))

    for i in range(taskNum):
        left=i*len(usersList)
        right=(i+1)*len(usersList)

        trueY=Y_label[left:right]
        trueY=np.where(trueY==1)[0]
        trueY=set(trueY)
        if len(trueY)==0:continue

        predictY=Y_predict[left:right]
        predictY=getTopK(predictY,k)
        #print("true",trueY)
        #print("predict",predictY)
        if len(trueY.intersection(predictY))>0:
            Y[i]=1

    return Y

#select top k users based on DIG
def topKDIGUsers(data,k):

    usersList=getUsers(data.tasktype)
    Y_label=data.testLabel

    dataRank=getScoreOfDIG(data.tasktype)
    taskids=data.taskids[:data.testPoint]

    taskNum=len(Y_label)//len(usersList)
    Y=np.zeros(shape=taskNum,dtype=np.int)
    print("top %d users from DIG for %d tasks(%d winners,%d users) "%(k,taskNum,np.sum(Y_label),len(usersList)))

    for i in range(taskNum):
        left=i*len(usersList)
        right=(i+1)*len(usersList)
        taskid=taskids[left]

        trueY=Y_label[left:right]
        trueY=np.where(trueY==1)[0]
        trueY=set(trueY)
        if len(trueY)==0:continue


        userRank=dataRank[taskid]["ranks"]
        predictY=userRank[:,0]
        predictY=np.array(predictY[:k],dtype=np.int)
        #print("true",trueY)
        #print("predict",predictY)
        if len(trueY.intersection(predictY))>0:
            Y[i]=1

    return Y


#top k acc based on hard classification
def topKAccuracyWithDIG(Y_predict2,data,k,reranking=False):
    '''
    :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                      else return false
    :param Y_predict2: a list of recommended entries
    :param data: the data set containing actual labels
    :return: Y, array with each element indicate the result of ground-truth
    '''
    print("using DIG,re-ranking=",reranking)
    # measure top k accuracy
    dataRank=getScoreOfDIG(data.tasktype)
    taskids=data.taskids[:data.testPoint]

    usersList=getUsers(data.tasktype)
    Y_label=data.testLabel

    taskNum=len(Y_label)//len(usersList)
    Y=np.zeros(shape=taskNum,dtype=np.int)

    for i in range(taskNum):
        left=i*len(usersList)
        right=(i+1)*len(usersList)
        taskid=taskids[left]

        userRank=dataRank[taskid]["ranks"]

        trueY=Y_label[left:right]
        trueY=np.where(trueY==1)[0]
        trueY=set(trueY)
        if len(trueY)==0:continue

        predictY=Y_predict2[left:right]
        predictY=np.where(predictY==1)[0]
        predictY=set(predictY[:k])

        if len(predictY)<k:
            #add users to meet requirement
            pos=0
            while len(predictY)<k and pos<len(userRank):
                predictY.add(int(userRank[pos][0]))
                pos+=1
        elif len(predictY)>k and reranking:
            #re-rank users using DIG
            t_predictY=[]
            for i in range(len(userRank)):
                rank_=int(userRank[i][0])
                if rank_ in predictY:
                    t_predictY.append(rank_)
                if len(t_predictY)>=k:
                    break

            predictY=t_predictY[:k]

        if len(trueY.intersection(predictY))>0:
            Y[i]=1

    return Y


def topKAccuracy(Y_predict2,data,k):
    # measure top k accuracy
    # batch data into task centered array
    print("without DIG")

    usersList=getUsers(data.tasktype)
    Y_label=data.testLabel

    taskNum=len(Y_label)//len(usersList)
    Y=np.zeros(shape=taskNum,dtype=np.int)

    for i in range(taskNum):
        left=i*len(usersList)
        right=(i+1)*len(usersList)

        trueY=Y_label[left:right]
        trueY=np.where(trueY==1)[0]
        trueY=set(trueY)
        if len(trueY)==0:continue

        predictY=Y_predict2[left:right]
        predictY=np.where(predictY==1)[0]
        predictY=set(predictY[:k])

        if len(trueY.intersection(predictY))>0:
            Y[i]=1

    return Y

#this method is to test topk acc when the submit status is known
def topKAccuracyOnSubset(Y_predict2,data,k):
    # measure top k accuracy
    # batch data into task centered array
    print("traditional assumption top k acc")

    submitlabels=data.submitLabelClassification[:data.testPoint]
    for p in range(len(submitlabels)):
        if submitlabels[p]==0:
            Y_predict2[p]=0

    usersList=getUsers(data.tasktype)
    Y_label=data.testLabel

    taskNum=len(Y_label)//len(usersList)
    Y=np.zeros(shape=taskNum,dtype=np.int)

    for i in range(taskNum):
        left=i*len(usersList)
        right=(i+1)*len(usersList)

        trueY=Y_label[left:right]
        trueY=np.where(trueY==1)[0]
        trueY=set(trueY)
        if len(trueY)==0:continue

        predictY=Y_predict2[left:right]
        predictY=getTopK(predictY,k)
        predictY=set(predictY[:k])

        if len(trueY.intersection(predictY))>0:
            Y[i]=1

    return Y
