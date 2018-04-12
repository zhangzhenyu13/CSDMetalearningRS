from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers

class CascadingModel:
    def __init__(self):
        self.regModel=None
        self.subModel=None
        self.winModel=None

    def loadModel(self,tasktype,models):

        self.regModel=models[0]()
        self.subModel=models[1]()
        self.winModel=models[2]()
        self.regModel.name=tasktype+"-classifier(Reg)"
        self.subModel.name=tasktype+"-classifier(Sub)"
        self.winModel.name=tasktype+"-classifier(Win)"
        self.regModel.loadModel()
        self.subModel.loadModel()
        self.winModel.loadModel()

        self.subExpr=getSubnumOfDIG(tasktype)
        self.scoreExpr=getScoreOfDIG(tasktype)
        self.users=getUsers(tasktype,mode=2)

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
                if regY[pos]==0:
                    continue

                #sub
                topN=int(0.5*len(self.users))
                selectedusers=reRankSubUsers(self.subExpr,taskid,topN)

                if subY[pos]==0 and j not in selectedusers:
                    continue

                #winner
                topN=int(0.2*len(self.users))
                selectedusers=reRankWinUsers(self.scoreExpr,taskid,topN)

                if winY[pos]==0 and j not in selectedusers:
                    continue
                Y[pos]=1

        return Y

