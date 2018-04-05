import pickle
import numpy as np
import time
from Utility.TagsDef import *
import threading
from collections import Counter
from imblearn import under_sampling,over_sampling

class DataSetTopcoder:
    def __init__(self,testratio=0.2,validateratio=0.1):
        self.dataSet=None
        self.trainX=None
        self.trainLabel=None
        self.testX=None
        self.testLabel=None
        self.validateX=None
        self.validateLabel=None
        self.testRatio=testratio
        self.validateRatio=validateratio

    def setParameter(self,tasktype,mode):
        #file data
        self.tasktype=tasktype.replace("/","_")
        self.filepath="../data/TopcoderDataSet/"+ModeTag[mode].lower()+"HistoryBasedData/"+self.tasktype+"-user_task.data"

    def indexDataPoint(self,taskids):
        id=taskids[0]
        IDIndex=[(id,0)]
        for i in range(1,len(taskids)):
            if taskids[i]!=id:
                id=taskids[i]
                IDIndex.append((id,i))
        return IDIndex

    def fetchData(self,files,key):
        '''
        :param files: the files that contain the data
        :param key: the data key
        :return: X, array like, containing the data
        '''
        print(self.tasktype+" fetching data,key="+key)
        VecX=[None for i in range(len(files))]
        self.fetchOne(files[0],key,0,VecX)
        X=VecX[0]
        for i in range(1,len(files)):
            file=files[i]
            self.fetchOne(file,key,i,VecX)
            X=np.concatenate((X,VecX[i]),axis=0)

        return X

    def fetchOne(self,file,key,indedx,vecX):
        '''
        fetch data segment from one file and put it into vecX
        :param file:
        :param key:
        :return:
        '''
    def loadData(self):
        print(self.tasktype,"loading data")
        with open(self.filepath,"rb") as f:
            self.dataSet=pickle.load(f)

        users=self.fetchData(self.dataSet,"users")
        tasks=self.fetchData(self.dataSet,"tasks")
        self.taskids=self.fetchData(self.dataSet,"taskids")

        X=np.concatenate((tasks,users),axis=1)

        self.IDIndex=self.indexDataPoint(taskids=self.taskids)
        tp=int(self.testRatio*len(self.IDIndex))
        self.testPoint=self.IDIndex[tp][1]
        vp=int((self.testRatio+self.validateRatio)*len(self.IDIndex))
        self.validatePoint=self.IDIndex[vp][1]


        self.trainX=X[self.validatePoint:]
        self.validateX=X[self.testPoint:self.validatePoint]
        self.testX=X[:self.testPoint]

        print("feature length for user(%d) and task(%d) is %d"%(len(users[0]),len(tasks[0]),len(X[0])))
        print("loaded all the instances, size=%d"%len(self.taskids),
              "test point=%d, validate point=%d"%(self.testPoint,self.validatePoint))

    def ReSampling(self,data,labels):

        label_status=Counter(labels)
        print(self.tasktype,"data "+self.tasktype,label_status)
        if len(label_status)<2:
            print(self.tasktype,"no need to resample")
            return data,labels
        n_samples=1e+10
        for k in label_status.keys():
            if label_status[k]<n_samples:
                n_samples=label_status[k]
        #print("n_neighbors<=",n_samples)
        try:
            resampling=over_sampling.ADASYN(n_neighbors=n_samples)

            data,labels=resampling.fit_sample(data,labels)
        except ValueError:
            print(self.tasktype,"resampling using random method")
            resampling=over_sampling.RandomOverSampler()
            data,labels=resampling.fit_sample(data,labels)

        label_status=Counter(labels)
        print(self.tasktype,"sampling status=",label_status)

        return data,labels



class TopcoderReg(DataSetTopcoder):
    def __init__(self,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)

    def fetchOne(self,file,key,index,vecX):
        t0=time.time()
        with open(file,"rb") as f:
            data=pickle.load(f)
            X=np.array(data[key])
        vecX[index]=X
        #print(X.shape)
        print(" fetched segment:",index,"in %ds"%(time.time()-t0))

    def RegisterRegressionData(self):
        Y=self.fetchData(self.dataSet,"regists")
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def RegisterClassificationData(self):
        self.RegisterRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel=np.array(self.testLabel>0,dtype=np.int)


class TopcoderSub(DataSetTopcoder):
    def __init__(self,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
    def fetchOne(self,file,key,index,vecX):
        t0=time.time()
        with open(file,"rb") as f:
            data=pickle.load(f)
            regists=np.array(data["regists"])
            X=np.array(data[key])
            indices=np.where(regists>0)[0]
            if len(indices)>0:
                X=X[indices]
        vecX[index]=X
        print(" fetched segment:",index,"in %ds"%(time.time()-t0))

    def CommitRegressionData(self):
        Y=self.fetchData(self.dataSet,"regists")
        indices=np.where(Y>0)[0]

        Y=self.fetchData(self.dataSet,"submits")

        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def CommitClassificationData(self):
        self.CommitRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel=np.array(self.testLabel>0,dtype=np.int)

class TopcoderWin(DataSetTopcoder):
    def __init__(self,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
    def fetchOne(self,file,key,index,vecX):
        t0=time.time()
        with open(file,"rb") as f:
            data=pickle.load(f)
            submits=np.array(data["submits"],dtype=np.int)
            X=np.array(data[key])
            indices=np.where(submits>0)[0]
            if len(indices)>0:
                X=X[indices]
        vecX[index]=X
        print(" fetched segment:",index,"in %ds"%(time.time()-t0))

    def WinRankData(self):
        Y=self.fetchData(self.dataSet,"ranks")
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def WinClassificationData(self):
        self.WinRankData()
        self.trainLabel=np.array(self.trainLabel==0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel==0,dtype=np.int)
        self.testLabel=np.array(self.testLabel==0,dtype=np.int)


