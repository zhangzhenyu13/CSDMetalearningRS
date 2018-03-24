import pickle
import numpy as np
import pandas as pd

class DataSetTopcoder:
    def __init__(self,splitratio=0.8,validateratio=0.1,filepath=None):
        self.dataSet=None
        self.trainX=None
        self.trainLabel=None
        self.testX=None
        self.testLabel=None
        self.validateX=None
        self.validateLabel=None
        self.splitRatio=splitratio
        self.validateRatio=validateratio
        #file data
        if filepath is not None:
            self.filepath=filepath
        self.loadData()

    def loadData(self):
        with open(self.filepath,"rb") as f:
            self.dataSet=pickle.load(f)
        users=self.dataSet["users"]
        tasks=self.dataSet["tasks"]
        X=np.concatenate((tasks,users),axis=1)
        self.trainSize=int(self.splitRatio*len(X))
        self.validateSize=int(self.validateRatio*self.trainSize)
        self.trainX=X[:self.trainSize-self.validateSize]
        self.validateX=X[self.trainSize-self.validateSize:self.trainSize]
        self.testX=X[self.trainSize:]
        print("feature length for user(%d) and task(%d) is %d"%(len(users[0]),len(tasks[0]),len(X[0])))
        print("loaded all the instances, size=%d"%len(X),"trainSize=%d, validateSize=%d"%(self.trainSize,self.validateSize))

    def CommitRegressionData(self):
        Y=np.array(self.dataSet["submits"])
        Y=Y[np.where(Y)]
        self.trainLabel=Y[:self.trainSize-self.validateSize]
        self.validateLabel=Y[self.trainSize-self.validateSize:self.trainSize]
        self.testLabel=Y[self.trainSize:]
    def CommitClassificationData(self):
        self.CommitRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel=np.array(self.testLabel>0,dtype=np.int)
    def WinRankData(self):
        Y=np.array(self.dataSet["ranks"])
        self.trainLabel=Y[:self.trainSize-self.validateSize]
        self.validateLabel=Y[self.trainSize-self.validateSize:self.trainSize]
        self.testLabel=Y[self.trainSize:]
    def WinClassificationData(self):
        self.WinRankData()
        self.trainLabel=np.array(self.trainLabel==0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel==0,dtype=np.int)
        self.testLabel=np.array(self.testLabel==0,dtype=np.int)

class DataSetTopcoderCluster:
    def __init__(self,splitraio=0.8,validateratio=0.1,filepath=None):
        self.dataSet=None
        self.trainX=None
        self.validateX=None
        self.testX=None
        self.trainLabel=None
        self.validateLabel=None
        self.testLabel=None
        self.splitRatio=splitraio
        self.validateRatio=validateratio
        #file data
        if filepath is not None:
            self.filepath=filepath
        self.loadData()

    def loadData(self):

        with open( self.filepath, "rb") as f:
            self.dataSet=pickle.load(f)
        self.clusternames=self.dataSet.keys()

        #with open("../data/saved_ML_models/clusteringModel" + str(choice) + ".pkl", "rb") as f:
        #    self.clusterModel=pickle.load(f)

    def loadClusters(self,clustername):
        self.activecluster=clustername
        data=self.dataSet[self.activecluster]

        users = data["users"]
        tasks = data["tasks"]
        if len(users)<20:
            self.trainSize=0
            return False
        X = np.concatenate((tasks, users), axis=1)
        self.trainSize=int(self.splitRatio*len(X))
        self.validateSize=int(self.validateRatio*self.trainSize)
        self.trainX=X[:self.trainSize-self.validateSize]
        self.validateX=X[self.trainSize-self.validateSize:self.trainSize]
        self.testX = X[self.trainSize:]

        print("data(%s) set size=%d, trainSize=%d, validateSize=%d"%(self.activecluster,len(X),self.trainSize,self.validateSize))

        return True

    def CommitRegressionData(self):
        data=self.dataSet[self.activecluster]
        Y=np.array(data["submits"])
        self.trainLabel=Y[:self.trainSize-self.validateSize]
        self.validateLabel=Y[self.trainSize-self.validateSize:self.trainSize]
        self.testLabel = Y[self.trainSize:]

    def CommitClassificationData(self):
        self.CommitRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel = np.array(self.testLabel>0,dtype=np.int)

    def WinRankData(self):
        data = self.dataSet[self.activecluster]
        Y=np.array(data["ranks"])
        self.trainLabel=Y[:self.trainSize-self.validateSize]
        self.validateLabel=Y[self.trainSize-self.validateSize:self.trainSize]
        self.testLabel = Y[self.trainSize:]

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
