import pickle
import numpy as np
import pandas as pd
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

def topKAccuracy(Y_predict,Y_true):
    '''
    :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                      else return false
    :param Y_predict:
    :param Y_true:
    :return: boolean
    '''
    Y=np.zeros(shape=(len(Y_predict)),dtype=np.bool)
    for i in range(len(Y)):
        tag=False
        for  ele in Y_predict:
            if ele in Y_true:
                tag=True
                break
        Y[i]=tag
    return Y

class classificationAssess:
    def __init__(self,filename):
        self.runpath = "../data/runResults/" + filename + ".txt"
        self.data = {"cluster": [], "train_acc": [], "test_acc": [],
                     "train_size": [], "test_size": [],
                     "recall":[],"precision":[],"model_name":[]}
    def processCFM(self,cfm,acc):
        if acc==1:
            precision=1
            recall=1
        else:
            if cfm[0][0]==0:
                if cfm[0][1]==0:
                    recall=1
                else:
                    recall=0
                if cfm[1][0]==0:
                    precision=1
                else:
                    precision=0
            else:
                recall=cfm[0][0]/(cfm[0][0]+cfm[0][1])
                precision=cfm[0][0]/(cfm[1][0]+cfm[0][0])
        return recall,precision
    def addValue(self,record):
        #print(record)
        self.data["cluster"].append(record[0])
        self.data["train_size"].append(record[1])
        self.data["test_size"].append(record[2])
        self.data["train_acc"].append(record[3])
        self.data["test_acc"].append(record[4])
        cfm=record[5]
        recall,precision=self.processCFM(cfm,record[4])
        self.data["precision"].append(precision)
        self.data["recall"].append(recall)
        self.data["model_name"].append(record[6])
    def saveData(self):
        self.data = pd.DataFrame(self.data,columns=["cluster","train_size","test_size","train_acc","test_acc","recall","precision"])
        self.data.to_csv(self.runpath)

class DataSetTopcoder:
    def __init__(self,splitratio=0.8,validateratio=0.1):
        self.dataSet=None
        self.trainX=None
        self.trainLabel=None
        self.testX=None
        self.testLabel=None
        self.validateX=None
        self.validateLabel=None
        self.splitRatio=splitratio
        self.validateRatio=validateratio
        self.loadData()

    def loadData(self,choice=1):
        with open("../data/Instances/task_user"+str(choice)+".data","rb") as f:
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

class DataSetTopCoderReg:
    def __init__(self,splitratio=0.8,validateratio=0.1):
        self.dataSet = None
        self.trainX = None
        self.trainLabel = None
        self.validateX=None
        self.validateLabel=None
        self.testX = None
        self.testLabel = None
        self.splitRatio = splitratio
        self.validateRatio=validateratio
        self.loadData()
    def loadData(self,choice=1):
        with open("../data/Instances/regsdata/task_userReg"+str(choice)+".data0_165","rb") as f:
            self.dataSet=pickle.load(f)
        users=self.dataSet["users"]
        tasks=self.dataSet["tasks"]
        regists=np.array(self.dataSet["regists"])
        X=np.concatenate((tasks,users),axis=1)
        self.trainSize=int(self.splitRatio*len(X))
        self.validateSize=int(self.validateRatio*self.trainSize)
        self.trainX=X[:self.trainSize-self.validateSize]
        self.validateX=X[self.trainSize-self.validateSize:self.trainSize]
        self.testX=X[self.trainSize:]
        self.trainLabel=regists[:self.trainSize-self.validateSize]
        self.validateLabel=regists[self.trainSize-self.validateSize:self.trainSize]
        self.testLabel=regists[self.trainSize:]
        print("feature length for user(%d) and task(%d) is %d"%(len(users[0]),len(tasks[0]),len(X[0])))
        print("loaded all the instances, size=%d"%len(X),"trainSize=%d, validateSize=%d"%(self.trainSize,self.validateSize))

class DataSetTopcoderCluster:
    def __init__(self,splitraio=0.8,validateratio=0.1):
        self.dataSet=None
        self.trainX=None
        self.validateX=None
        self.testX=None
        self.trainLabel=None
        self.validateLabel=None
        self.testLabel=None
        self.splitRatio=splitraio
        self.validateRatio=validateratio
        self.loadData()

    def loadData(self,choice=1):

        with open("../data/Instances/task_user_local" + str(choice) + ".data" , "rb") as f:
            self.dataSet=pickle.load(f)
        self.clusternames=self.dataSet.keys()

        #with open("../data/saved_ML_models/clusteringModel" + str(choice) + ".pkl", "rb") as f:
        #    self.clusterModel=pickle.load(f)

    def loadClusters(self,clustername):
        self.activecluster=clustername
        data=self.dataSet[self.activecluster]

        users = data["users"]
        tasks = data["tasks"]
        X = np.concatenate((tasks, users), axis=1)
        self.trainSize=int(self.splitRatio*len(X))
        self.validateSize=int(self.validateRatio*self.trainSize)
        self.trainX=X[:self.trainSize-self.validateSize]
        self.validateX=X[self.trainSize-self.validateSize:self.trainSize]
        self.testX = X[self.trainSize:]

        print("data(%s) set size=%d, trainSize=%d, validateSize=%d"%(self.activecluster,len(X),self.trainSize,self.validateSize))

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
