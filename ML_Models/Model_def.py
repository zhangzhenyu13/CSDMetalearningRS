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

    def findPath(self):
        modelpath="../data/saved_ML_models/"+self.name+".pkl"
        return modelpath
    def loadModel(self):
        with open(self.findPath(),"rb") as f:
            data=pickle.load(f)
            self.model=data["model"]
            self.name=data["name"]
    def saveModel(self):
        with open(self.findPath(),"wb") as f:
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



