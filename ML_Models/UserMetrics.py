import numpy as np
from Utility.TagsDef import getUsers
import pickle
from Utility.personalizedSort import MySort

#metrics

class TopKMetrics:
    def __init__(self,tasktype):
        self.subRank=self.__getSubnumOfDIG(tasktype)
        self.scoreRank=self.__getScoreOfDIG(tasktype)
    #subnum based rank data
    def __getSubnumOfDIG(self,tasktype):
        with open("../data/UserInstances/UserGraph/SubNumBased/"+tasktype+"-UserInteraction.data","rb") as f:
            dataRank=pickle.load(f)
        return dataRank

    #score based rank data
    def __getScoreOfDIG(self,tasktype):
        with open("../data/UserInstances/UserGraph/ScoreBased/"+tasktype+"-UserInteraction.data","rb") as f:
            rankData=pickle.load(f)
        return rankData

    #clip indices of top k data from given vector X
    def __getTopKonPossibility(self,P,k):
        x_vec=[]
        for i in range(len(P)):
            x_vec.insert(i,[i,P[i]])

        mysort=MySort(x_vec)
        mysort.compare_vec_index=-1
        x_vec=mysort.mergeSort()
        x_vec=np.array(x_vec)[:k]

        return np.array(x_vec[:,0],dtype=np.int),np.array(x_vec[:,1])

    #clip indices of top k data from DIG
    def getTopKonDIGRank(self,userRank,k):
        predictY=userRank[:,0]
        predictY=np.array(predictY[:k],dtype=np.int)
        return predictY,np.array(userRank[:,1])

    #clip indices of top k data from weighted sum of P and R
    def __getTopKonPDIGScore(self,predictP,predictR,rank_weight=0.5):
        Y=[]
        for i in range(len(predictP)):
            P=predictP[i]
            index=np.where(predictR[:,0]==P[0])[0]

            if len(index)>0:
                R=predictR[index]
                Y.append([P[0],rank_weight*P[1]+(1-rank_weight)*R[1]])
                predictP=np.delete(predictP,i,axis=0)
                predictR=np.delete(predictR,index,axis=0)

        for P in predictP:
            Y.append([P[0],rank_weight*P[1]])
        for R in predictR:
            Y.append([R[0],(1-rank_weight)*R[1]])
        #sort Y
        ms=MySort(Y)
        ms.compare_vec_index=-1
        Y=ms.mergeSort()
        Y=np.array(Y)
        return np.array(Y[:,0],dtype=np.int),np.array(Y[:,1])

    #select top k users based on its prediction possibility
    def topKPossibleUsers(self,Y_predict,data,k):

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
            predictY,_ =self.__getTopKonPossibility(predictY,k)
            #print("true",trueY)
            #print("predict",predictY)
            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y

    #select top k users based on DIG
    def topKDIGUsers(self,data,k):

        usersList=getUsers(data.tasktype)
        Y_label=data.testLabel

        dataRank=self.scoreRank
        taskids=data.taskids[:data.testPoint]

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        print("top %d users from DIG for %d tasks(%d winners,%d users) "%
              (k,taskNum,np.sum(Y_label),len(usersList)))

        for i in range(taskNum):
            left=i*len(usersList)
            right=(i+1)*len(usersList)
            taskid=taskids[left]

            trueY=Y_label[left:right]
            trueY=np.where(trueY==1)[0]
            trueY=set(trueY)
            if len(trueY)==0:continue

            userRank=dataRank[taskid]["ranks"]
            predictY,_ =self.getTopKonDIGRank(userRank,k)
            #print("true",trueY)
            #print("predict",predictY)
            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y


    #top k acc based on hard classification
    def topKPDIGUsers(self,Y_predict2,data,k,rank_weight=0.5):
        '''
        :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                          else return false
        :param Y_predict2: a list of recommended entries
        :param data: the data set containing actual labels
        :return: Y, array with each element indicate the result of ground-truth
        '''
        print("",rank_weight)

        # measure top k accuracy
        dataRank=self.scoreRank
        taskids=data.taskids[:data.testPoint]

        usersList=getUsers(data.tasktype)
        Y_label=data.testLabel

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        print("top %d users from Possibility&DIG(using DIG,re-ranking=%f) for %d tasks(%d winners,%d users)"%
              (k,taskNum,rank_weight,np.sum(Y_label),len(usersList)))

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
            predictP=self.__getTopKonPossibility(predictY,k)
            predictR=self.getTopKonDIGRank(userRank,k)
            predictY,_ =self.__getTopKonPDIGScore(predictP,predictR,rank_weight)
            predictY=predictY[:k]
            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y


    #this method is to test topk acc when the submit status is known
    def topKSUsers(self,Y_predict2,data,k):
        # measure top k accuracy
        # batch data into task centered array
        print("status observed assumption top k acc")

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
            predictY, _=self.__getTopKonPossibility(predictY,k)
            predictY=set(predictY[:k])

            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y
