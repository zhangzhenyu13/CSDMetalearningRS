import _pickle as pickle

class ML_model:
    def __init__(self):
        self.model=None
        self.name=""
        self.threshold=0.5
        self.verbose=1
    def predict(self,X):
        '''
        predict the result based on given X
        :param X: input samples,(n,D)
        :return: given result, class or a real num
        '''

    def trainModel(self,dataSet):
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
            pickle.dump(data,f,True)

