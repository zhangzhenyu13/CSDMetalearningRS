import numpy as np
from Utility.TagsDef import getUsers
import pickle,copy
from Utility.personalizedSort import MySort
from sklearn import preprocessing
#metrics

class TopKMetrics:
    mmscaler=preprocessing.MinMaxScaler(feature_range=(0,1))

    def __init__(self,tasktype,callall=False,verbose=1,testMode=False):
        self.verbose=verbose
        self.callall=callall
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
    def getMRR(self,trueY,predictY):
        x_vec=[]
        for i in range(len(predictY)):
            x_vec.insert(i,[i,predictY[i]])

        mysort=MySort(x_vec)
        mysort.compare_vec_index=-1
        x_vec=mysort.mergeSort()
        x_vec=np.array(x_vec)

        predictY=np.array(x_vec[:,0],dtype=np.int)

        mrr=0
        for i in range(len(predictY)):
            if predictY[i] in trueY:
                mrr=1.0/(i+1.0)
                break
        return mrr

    #clip indices of top k data from DIG
    def getTopKonDIGRank(self,userRank,k):

        return np.array(userRank[:k,0],dtype=np.int),np.array(userRank[:k,1])

    #clip indices of top k data from weighted sum of P and R
    def getTopKonPDIGScore(self,predictP,predictR,rank_weight):
        Y=[]
        indexP=predictP[0]
        indexR=predictR[0]
        scoreP=predictP[1]
        scoreR=predictR[1]
        n_all=len(indexP)

        selUsers=list(set(indexP).intersection(indexR))
        if len(selUsers)==len(indexP):
            return indexP,scoreP

        if len(selUsers)!=0:
            for index in selUsers:
                PI=np.where(indexP==index)[0]
                RI=np.where(indexR==index)[0]
                Y.append([indexP[RI[0]],scoreP[RI[0]]])
                indexP=np.delete(indexP,PI,axis=0)
                scoreP=np.delete(scoreP,PI,axis=0)
                indexR=np.delete(indexR,RI,axis=0)
                scoreR=np.delete(scoreR,RI,axis=0)
        n_left=n_all-len(selUsers)
        n_r=int((1-rank_weight)*n_left)
        n_p=n_left-n_r
        for i in range(n_p):
            Y.append([indexP[i],scoreP[i]])
        for i in range(n_r):
            Y.append([indexR[i],scoreR[i]])

        Y=np.array(Y)

        return np.array(Y[:,0],dtype=np.int),np.array(Y[:,1])

    #select top k users based on its prediction possibility
    def topKPossibleUsers(self,Y_predict,Y_label,k):
        Y_predict2=copy.deepcopy(Y_predict)

        usersList=self.userlist

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        if self.verbose>0:
            print("top %d users from possibility for %d tasks(%d winners,%d users) "%
              (k,taskNum,np.sum(Y_label),len(usersList)))

        for i in range(taskNum):
            left=i*len(usersList)
            right=(i+1)*len(usersList)

            trueY=Y_label[left:right]
            trueY=np.where(trueY==1)[0]
            trueY=set(trueY)
            if len(trueY)==0:continue

            predictY=Y_predict2[left:right]
            predictY,_ =self.getTopKonPossibility(predictY,k)
            if self.verbose>1:
                print("true",trueY)
                print("predict",predictY)

            containUsers=trueY.intersection(predictY)
            if len(containUsers)>0:
                Y[i]=1
                if self.callall:
                    Y[i]=len(containUsers)/len(trueY)

        return Y

    #select top k users based on its prediction possibility
    def getAllMRR(self,Y_predict,Y_label):
        Y_predict2=copy.deepcopy(Y_predict)

        usersList=self.userlist

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.float32)
        if self.verbose>0:
            print("MRR for %d tasks(%d winners,%d users) "%
              (taskNum,np.sum(Y_label),len(usersList)))

        for i in range(taskNum):
            left=i*len(usersList)
            right=(i+1)*len(usersList)

            trueY=Y_label[left:right]
            trueY=np.where(trueY==1)[0]
            trueY=set(trueY)
            if len(trueY)==0:continue

            predictY=Y_predict2[left:right]

            mrr=self.getMRR(trueY,predictY)
            Y[i]=mrr

        return Y


    #select top k users based on DIG
    def topKDIGUsers(self,Y_label,taskids,k):

        usersList=self.userlist

        dataRank=self.scoreRank

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        if self.verbose>0:
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
            if self.verbose>1:
                print("true",trueY)
                print("predict",predictY)

            containUsers=trueY.intersection(predictY)
            if len(containUsers)>0:
                Y[i]=1
                if self.callall:
                    Y[i]=len(containUsers)/len(trueY)

        return Y

    #top k acc based on hard classification
    def topKPDIGUsers(self,Y_predict,Y_label,taskids,k,rank_weight):
        '''
        :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                          else return false
        :param Y_predict2: a list of recommended entries
        :param data: the data set containing actual labels
        :return: Y, array with each element indicate the result of ground-truth
        '''

        Y_predict2=copy.deepcopy(Y_predict)
        # measure top k accuracy
        dataRank=self.scoreRank

        usersList=self.userlist

        taskNum=len(Y_label)//len(usersList)
        Y=np.zeros(shape=taskNum,dtype=np.int)
        if self.verbose>0:
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
            if self.verbose>1:
                print("true",trueY)
                print("predict",predictY)

            containUsers=trueY.intersection(predictY)
            if len(containUsers)>0:
                Y[i]=1
                if self.callall:
                    Y[i]=len(containUsers)/len(trueY)

        return Y

    #this method is to test topk acc when the submit status is known
    def topKSUsers(self,Y_predict,Y_label,Y_sublabel,k):
        # measure top k accuracy
        # batch data into task centered array
        if self.verbose>0:
            print("sub status observed assumption top k acc")

        Y_predict2=copy.deepcopy(Y_predict)
        submitlabels=Y_sublabel
        for p in range(len(submitlabels)):
            if submitlabels[p]==0:
                Y_predict2[p]=0

        usersList=self.userlist


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
            if self.verbose>1:
                print("trueY",trueY)
                print("predictY",predictY)

            containUsers=trueY.intersection(predictY)
            if len(containUsers)>0:
                Y[i]=1
                if self.callall:
                    Y[i]=len(containUsers)/len(trueY)

        return Y
    #this method is to test topk acc when the register status is known
    def topKRUsers(self,Y_predict,Y_label,Y_reglabel,k):
        # measure top k accuracy
        # batch data into task centered array
        if self.verbose>0:
            print("reg status observed assumption top k acc")
        Y_predict2=copy.deepcopy(Y_predict)
        reglabels=Y_reglabel
        for p in range(len(reglabels)):
            if reglabels[p]==0:
                Y_predict2[p]=0

        usersList=self.userlist

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
            if self.verbose>1:
                print("trueY",trueY)
                print("predictY",predictY)

            containUsers=trueY.intersection(predictY)
            if len(containUsers)>0:
                Y[i]=1
                if self.callall:
                    Y[i]=len(containUsers)/len(trueY)
        return Y
