import pickle
import numpy as np
import gc
from Utility.TagsDef import *

from collections import Counter
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

    def loadData(self):
        print(self.tasktype,"loading data")
        with open(self.filepath,"rb") as f:
            self.dataSet=pickle.load(f)

        users=self.fetchData(self.dataSet,"users")
        tasks=self.fetchData(self.dataSet,"tasks")
        taskids=self.fetchData(self.dataSet,"taskids")

        X=np.concatenate((tasks,users),axis=1)

        self.IDIndex=self.indexDataPoint(taskids=taskids)
        tp=int(self.testRatio*len(self.IDIndex))
        self.testPoint=self.IDIndex[tp][1]
        vp=int((self.testRatio+self.validateRatio)*len(self.IDIndex))
        self.validatePoint=self.IDIndex[vp][1]


        self.trainX=X[self.validatePoint:]
        self.validateX=X[self.testPoint:self.validatePoint]
        self.testX=X[:self.testPoint]

        print("feature length for user(%d) and task(%d) is %d"%(len(users[0]),len(tasks[0]),len(X[0])))
        print("loaded all the instances, size=%d"%len(taskids),
              "test point=%d, validate point=%d"%(self.testPoint,self.validatePoint))

    def ReSampling(self,data,labels,method):
        resampling=method()
        label_status=Counter(labels)
        print("data "+self.tasktype,label_status)
        data,labels=resampling.fit_sample(data,labels)
        label_status=Counter(labels)
        print("sampling status=",label_status)

        return data,labels



class TopcoderReg(DataSetTopcoder):
    def __init__(self,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
    def fetchData(self,files,key):
        print(self.tasktype+" fetching registration data,key="+key)
        with open(files[0],"rb") as f:
            data=pickle.load(f)
            X=np.array(data[key])

        if len(files)<2:
            return X

        for file in files[1:]:
            with open(file,"rb") as f:
                data=pickle.load(f)
                X=np.concatenate((X,data[key]),axis=0)
        data=None
        gc.collect()
        return X

    def RegisterRegressionData(self):
        Y=self.fetchData(self.dataSet,"regists")
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def RegisterClassificationData(self):
        self.RegisterRegressionData()
        self.trainLabel=np.array(self.trainLabel==0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel==0,dtype=np.int)
        self.testLabel=np.array(self.testLabel==0,dtype=np.int)


class TopcoderSub(DataSetTopcoder):
    def __init__(self,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
    def fetchData(self,files,key):
        print(self.tasktype+" fetching submission data,key="+key)
        with open(files[0],"rb") as f:
            data=pickle.load(f)
            regists=np.array(data["regists"])
            X=np.array(data[key])
            indices=np.where(regists>0)[0]
            if len(indices)>0:
                X=X[indices]
        if len(files)<2:
            return X

        for file in files[1:]:
            with open(file,"rb") as f:
                data=pickle.load(f)
                regists=np.array(data["regists"])
                X=np.concatenate((X,data[key]),axis=0)
                indices=np.where(regists>0)[0]
                if len(indices)>0:
                    X=X[indices]
        data=None
        gc.collect()
        return X

    def CommitRegressionData(self):
        Y=self.fetchData(self.dataSet,"regists")
        indices=np.where(Y>0)[0]

        Y=self.fetchData(self.dataSet,"submits")

        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def CommitClassificationData(self):
        self.CommitRegressionData()
        self.trainLabel=np.array(self.trainLabel==0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel==0,dtype=np.int)
        self.testLabel=np.array(self.testLabel==0,dtype=np.int)

class TopcoderWin(DataSetTopcoder):
    def __init__(self,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)

    def fetchData(self,files,key):
        print(self.tasktype+" fetching winning data,key="+key)
        with open(files[0],"rb") as f:
            data=pickle.load(f)
            submits=np.array(data["submits"],dtype=np.int)
            X=np.array(data[key])
            indices=np.where(submits>0)[0]
            if len(indices)>0:
                X=X[indices]
        if len(files)<2:
            return X

        for file in files[1:]:
            with open(file,"rb") as f:
                data=pickle.load(f)
                submits=np.array(data["submits"],dtype=np.int)
                X=np.concatenate((X,data[key]),axis=0)
                indices=np.where(submits>0)[0]
                if len(indices)>0:
                    X=X[indices]
        data=None
        gc.collect()
        return X

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


