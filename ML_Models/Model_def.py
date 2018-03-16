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
        Y=np.array(Y>0,dtype=np.int)
        self.trainLabel=Y[:self.trainSize]
        self.testLabel=Y[self.trainSize:]
    def WinRankData(self):
        Y=np.array(self.dataSet["ranks"])
        self.trainLabel=Y[:self.trainSize]
        self.testLabel=Y[self.trainSize:]
    def WinClassificationData(self):
        Y=np.array(self.dataSet["ranks"])
        Y=np.array(Y==0,dtype=np.int)
        self.trainLabel=Y[:self.trainSize]
        self.testLabel=Y[self.trainSize:]

class DataSetTopcoderCluster:
    def __init__(self):
        self.dataSetTrain=None
        self.dataSetTest=None
        self.trainX=None
        self.testX=None
        self.trainLabel=None
        self.testLabel=None
        self.n_clusters=20
    def loadData(self,choice=1):

        with open("../data/Instances/task_user_local_train" + str(choice) + ".data" , "rb") as f:
            self.dataSetTrain=pickle.load(f)
        with open("../data/Instances/task_user_local_test" + str(choice) + ".data", "rb") as f:
            self.dataSetTest=pickle.load(f)
        with open("../data/clusterResult/kmeans" + str(choice) + ".pkl", "rb") as f:
            self.KM=pickle.load(f)

    def loadClusters(self,k_no):
        self.k_no=k_no

        data=self.dataSetTrain[self.k_no]
        users = data["users"]
        tasks = data["tasks"]
        X = np.concatenate((tasks, users), axis=1)
        self.trainX=X

        data=self.dataSetTest[self.k_no]
        users = data["users"]
        tasks = data["tasks"]
        X = np.concatenate((tasks, users), axis=1)
        self.testX = X

    def CommitRegressionData(self):
        data=self.dataSetTrain[self.k_no]
        Y=np.array(data["submits"])
        self.trainLabel=Y

        data = self.dataSetTest[self.k_no]
        Y = np.array(data["submits"])
        self.testLabel = Y

    def CommitClassificationData(self):
        self.CommitRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.testLabel = np.array(self.testLabel>0,dtype=np.int)

    def WinRankData(self):
        data = self.dataSetTrain[self.k_no]
        Y=np.array(data["ranks"])
        self.trainLabel=Y

        data = self.dataSetTest[self.k_no]
        Y = np.array(data["ranks"])
        self.testLabel = Y

    def WinClassificationData(self):
        self.WinRankData()
        self.trainLabel=np.array(self.trainLabel==0,dtype=np.int)
        self.testLabel=np.array(self.testLabel==0,dtype=np.int)
