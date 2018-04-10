from ML_Models.TraditionalModel import TraditionalClassifier
import numpy as np

class CascadingModel:
    def __init__(self):
        self.regModel=None
        self.subModel=None
        self.winModel=None

    def loadModel(self,tasktype):
        self.regModel=TraditionalClassifier()
        self.subModel=TraditionalClassifier()
        self.winModel=TraditionalClassifier()
        self.regModel.name=tasktype+"-classifier(Reg)"
        self.subModel.name=tasktype+"-classifier(Sub)"
        self.winModel.name=tasktype+"-classifier(Win)"
        self.regModel.loadModel()
        self.subModel.loadModel()
        self.winModel.loadModel()

    def predict(self,X):
        print("Cascading Model is predicting")
        regY=self.regModel.predict(X)
        subY=self.subModel.predict(X)
        winY=self.winModel.predict(X)
        Y=np.zeros(shape=len(X))

        for i in range(len(X)):
            if regY[i]==0:
                continue
            if subY[i]==0:
                continue
            if winY[i]==0:
                continue
            Y[i]=1

        return Y

