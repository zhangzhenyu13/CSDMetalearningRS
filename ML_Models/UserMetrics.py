import numpy as np
from Utility.TagsDef import getUsers
import pickle
from Utility.personalizedSort import MySort
from sklearn import preprocessing
#metrics

class TopKMetrics:
    def __init__(self,tasktype,verbose=1,testMode=False):
        self.verbose=verbose
        self.scoreRank=self.__getScoreOfDIG(tasktype)
        self.subRank=None
        if "#" in tasktype:
            pos=tasktype.find("#")
            self.subRank=self.__getSubnumOfDIG(tasktype[:pos])
        else:
            self.subRank=self.__getSubnumOfDIG(tasktype)
        if testMode:
            self.userlist=getUsers(tasktype+"-test",2)
        else:
            self.userlist=getUsers(tasktype,2)

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
    def getTopKonPossibility(self,P,k):
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

        return np.array(userRank[:k,0],dtype=np.int),np.array(userRank[:k,1])

    #clip indices of top k data from weighted sum of P and R
    def getTopKonPDIGScore(self,predictP,predictR,rank_weight=0.8):
        Y=[]
        indexP=predictP[0]
        indexR=predictR[0]
        scoreP=predictP[1]
        scoreR=predictR[1]

        maxR=np.max(scoreR)
        minR=np.min(scoreR)
        if maxR-minR>0:
            scoreR=(scoreR-minR)/(maxR-minR)
        #sort
        n_users=len(indexP)
        rmP=[]
        rmR=[]
        for i in range(n_users):

            index=np.where(indexR==indexP[i])[0]

            if len(index)>0:
                index=index[0]
                Y.append([indexP[i],rank_weight*scoreP[i]+(1-rank_weight)*scoreR[index]])
                rmP.append(i)
                rmR.append(index)

        for i in range(n_users):
            if i not in rmP:
                Y.append([indexP[i],rank_weight*scoreP[i]])
        for i in range(n_users):
            if i not in rmR:
                Y.append([indexR[i],(1-rank_weight)*scoreR[i]])
        #sort Y
        ms=MySort(Y)
        ms.compare_vec_index=-1
        Y=ms.mergeSort()
        Y=np.array(Y)
        return np.array(Y[:,0],dtype=np.int),np.array(Y[:,1])

    #select top k users based on its prediction possibility
    def topKPossibleUsers(self,Y_predict,data,k):

        usersList=self.userlist
        Y_label=data.testLabel

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        if self.verbose==1:
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
            predictY,_ =self.getTopKonPossibility(predictY,k)
            #print("true",trueY)
            #print("predict",predictY)
            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y

    #select top k users based on DIG
    def topKDIGUsers(self,data,k):

        usersList=self.userlist
        Y_label=data.testLabel

        dataRank=self.scoreRank
        taskids=data.taskids[:data.testPoint]

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        if self.verbose==1:
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
    def topKPDIGUsers(self,Y_predict2,data,k,rank_weight=0.9):
        '''
        :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                          else return false
        :param Y_predict2: a list of recommended entries
        :param data: the data set containing actual labels
        :return: Y, array with each element indicate the result of ground-truth
        '''

        # measure top k accuracy
        dataRank=self.scoreRank
        taskids=data.taskids[:data.testPoint]

        usersList=self.userlist
        Y_label=data.testLabel

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        if self.verbose==1:
            print("top %d users from Possibility&DIG(using DIG,re-ranking=%f) for %d tasks(%d winners,%d users)"%
              (k,rank_weight,taskNum,np.sum(Y_label),len(usersList)))

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
            predictP=self.getTopKonPossibility(predictY,k)
            predictR=self.getTopKonDIGRank(userRank,k)
            predictY,_ =self.getTopKonPDIGScore(predictP,predictR,rank_weight)
            predictY=predictY[:k]
            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y


    #this method is to test topk acc when the submit status is known
    def topKSUsers(self,Y_predict2,data,k):
        # measure top k accuracy
        # batch data into task centered array
        if self.verbose==1:
            print("sub status observed assumption top k acc")

        submitlabels=data.submitLabelClassification[:data.testPoint]
        for p in range(len(submitlabels)):
            if submitlabels[p]==0:
                Y_predict2[p]=0

        usersList=self.userlist
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
            predictY, _=self.getTopKonPossibility(predictY,k)
            predictY=set(predictY[:k])

            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y
    #this method is to test topk acc when the register status is known
    def topKRUsers(self,Y_predict2,data,k):
        # measure top k accuracy
        # batch data into task centered array
        if self.verbose==1:
            print("reg status observed assumption top k acc")

        reglabels=data.registerLabelClassification[:data.testPoint]
        for p in range(len(reglabels)):
            if reglabels[p]==0:
                Y_predict2[p]=0

        usersList=self.userlist
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
            predictY, _=self.getTopKonPossibility(predictY,k)
            predictY=set(predictY[:k])

            if len(trueY.intersection(predictY))>0:
                Y[i]=1

        return Y
