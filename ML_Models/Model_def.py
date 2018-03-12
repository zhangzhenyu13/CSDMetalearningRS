import pickle
import json
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
    def saveModel(self):
        with open("../data/saved_ML_models/"+self.name+".pkl","wb") as f:
            data={}
            data["model"]=self.model
            pickle.dump(data,f)

class DataSetTopcoder:
    def __init__(self):
        self.trainX=None
        self.trainLabel=None
        self.testX=None
        self.testLabel=None
    def loadData(self,cluster_no):
        pass