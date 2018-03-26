import pickle
import numpy as np
import gc

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

    def setParameter(self,tasktype,choice):
        #file data
        self.tasktype=tasktype.replace("/","_")
        self.filepath="../data/TopcoderDataSet/winHistoryBasedData/"+self.tasktype+"-user_task-"+str(choice)+".data"

    def indexDataPoint(self,taskids):
        id=taskids[0]
        IDIndex=[(id,0)]
        for i in range(1,len(taskids)):
            if taskids[i]!=id:
                id=taskids[i]
                IDIndex.append((id,i))
        return IDIndex

    def fetchData(self,files,key):
        with open(files[0],"rb") as f:
            data=pickle.load(f)
            X=np.array(data[key])
        del files[0]
        for file in files:
            with open(file,"rb") as f:
                data=pickle.load(f)
                X=np.concatenate((X,data["users"]),axis=0)
        data=None
        gc.collect()
        return X

    def loadData(self):
        with open(self.filepath,"rb") as f:
            self.dataSet=pickle.load(f)

        users=self.fetchData(self.dataSet,"users")
        tasks=self.fetchData(self.dataSet,"tasks")
        taskids=self.fetchData(self.dataSet,"taskids")

        X=np.concatenate((tasks,users),axis=1)

        self.IDIndex=self.indexDataPoint(taskids)
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

    def CommitRegressionData(self):
        Y=self.fetchData(self.dataSet,"submits")
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def CommitClassificationData(self):
        self.CommitRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel=np.array(self.testLabel>0,dtype=np.int)

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


if __name__ == '__main__':
    data = DataSetTopcoder()
    data.CommitClassificationData()
    from pandas import DataFrame as frame
    data_train = frame(data.trainX)
    data_train["Class"] = data.trainLabel
    data_train.to_csv("trainData.csv")
    data_test = frame(data.testX)
    data_test["Class"] = data.testLabel
    data_test.to_csv("testData.csv")
