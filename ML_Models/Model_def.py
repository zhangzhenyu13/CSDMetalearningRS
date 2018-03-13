import pickle
import numpy as np
class ML_model:
    def __init__(self):
        self.model=None
        self.name=""
        self.dataSet=None
    def predict(self,X):
        '''
        predict the result based on given X
        :param X: input samples,(n,D)
        :return: given result, class or a real num
        '''
    def trainModel(self):
        pass
    def loadModel(self):
        with open("../data/saved_ML_models/"+self.name+".pkl","rb") as f:
            data=pickle.load(f)
            self.model=data["model"]
            self.name=data["name"]
    def saveModel(self):
        with open("../data/saved_ML_models/"+self.name+".pkl","wb") as f:
            data={}
            data["model"]=self.model
            data["name"]=self.name
            pickle.dump(data,f)

class DataSetTopcoder:
    def __init__(self):
        self.dataSet=None
        self.trainX=None
        self.trainLabel=None
        self.testX=None
        self.testLabel=None
        self.splitRatio=0.8
        self.loadData()

    def loadData(self,choice=1):
        with open("../data/Instances/task_user"+str(choice)+".data","rb") as f:
            self.dataSet=pickle.load(f)
        users=self.dataSet["users"]
        tasks=self.dataSet["tasks"]
        X=np.concatenate((tasks,users),axis=1)
        self.trainSize=int(self.splitRatio*len(X))
        self.trainX=X[:self.trainSize]
        self.testX=X[self.trainSize:]
        print("feature length for user(%d) and task(%d) is %d"%(len(users[0]),len(tasks[0]),len(X[0])))
        print("loaded all the instances, size=%d"%len(X),"trainSize=%d"%self.trainSize)

    def CommitRegressionData(self):
        Y=np.array(self.dataSet["submits"])
        self.trainLabel=Y[:self.trainSize]
        self.testLabel=Y[self.trainSize:]
    def CommitClassificationData(self):
        Y=np.array(self.dataSet["submits"])
        Y=np.array(Y,dtype=np.int)
        self.trainLabel=Y[:self.trainSize]
        self.testLabel=Y[self.trainSize:]
    def WinRankData(self):
        Y=np.array(self.dataSet["ranks"])
        self.trainLabel=Y[:self.trainSize]
        self.testLabel=Y[self.trainSize:]
