from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers

class CascadingModel:
    def __init__(self):
        self.regModel=None
        self.subModel=None
        self.winModel=None

    def loadModel(self,tasktype,models):
        print(len(models),models)
        self.regModel=models[0]()
        self.subModel=models[1]()
        self.winModel=models[2]()
        self.regModel.name=tasktype+"-classifier(Reg)"
        self.subModel.name=tasktype+"-classifier(Sub)"
        self.winModel.name=tasktype+"-classifier(Win)"
        self.regModel.loadModel()
        self.subModel.loadModel()
        self.winModel.loadModel()

        self.mymetric=TopKMetrics(tasktype)
        self.subExpr=self.mymetric.subRank
        self.scoreExpr=self.mymetric.scoreRank
        self.users=getUsers(tasktype,mode=2)
        self.threshold=self.winModel.threshold

    def predict(self,X,taskids):
        print("Cascading Model is predicting")
        regY=self.regModel.predict(X)
        subY=self.subModel.predict(X)
        winY=self.winModel.predict(X)
        Y=np.zeros(shape=len(X))
        taskNum=len(X)//len(self.users)

        for i in range(taskNum):
            for j in range(len(self.users)):
                pos=i*len(self.users)+j
                taskid=taskids[pos]
                #reg
                if regY[pos]<self.threshold:
                    continue

                #sub
                topN=int(0.5*len(self.users))
                selectedusers,_ =self.mymetric.getTopKonDIGRank(self.subExpr[taskid]["ranks"],topN)

                if subY[pos]<self.threshold and j not in selectedusers:
                    continue

                #winner

                Y[pos]=winY[pos]

        return Y

