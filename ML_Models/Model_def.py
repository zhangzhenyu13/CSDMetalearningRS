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

    def findPath(self):
        modelpath="../data/saved_ML_models/classifiers/"+self.name+".pkl"
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




