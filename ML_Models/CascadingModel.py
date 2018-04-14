from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers
import json

class CascadingModel:
    def __init__(self):
        self.regModel=None
        self.subModel=None
        self.winModel=None
        self.regThreshold=0.3
        self.subThreshold=0.3
        self.winThreshold=0.5

    def loadModel(self,tasktype,models):
        print(len(models),models)
        self.regModel=models[0]()
        self.subModel=models[1]()
        self.winModel=models[2]()
        self.regModel.name=tasktype+"-classifierReg"
        self.subModel.name=tasktype+"-classifierSub"
        self.winModel.name=tasktype+"-classifierWin"
        self.regModel.loadModel()
        self.subModel.loadModel()
        self.winModel.loadModel()
        self.name=tasktype+"-classifierRules"

        self.mymetric=TopKMetrics(tasktype)
        self.subExpr=self.mymetric.subRank
        self.scoreExpr=self.mymetric.scoreRank
        self.users=getUsers(tasktype,mode=2)
        self.threshold=self.winModel.threshold
        try:
            self.loadConf()
        except:
            print("meta models config loading failed")

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
                if regY[pos]<self.regThreshold:
                    continue

                #sub
                topN=int(0.5*len(self.users))
                selectedusers,_ =self.mymetric.getTopKonDIGRank(self.subExpr[taskid]["ranks"],topN)

                if subY[pos]<self.subThreshold and j not in selectedusers:
                    continue
                #winner

                Y[pos]=winY[pos]

        return Y

    def saveConf(self):
        params={"regThreshold":self.regThreshold,"subThreshold":self.subThreshold,"winThreshold":self.winThreshol}
        with open("../data/saved_ML_models/MetaPredictor/"+self.name+".json","w") as f:
            json.dump(params,f)

    def loadConf(self):
        with open("../data/saved_ML_models/MetaPredictor/"+self.name+".json","w") as f:
            params=json.load(f)
        self.regThreshold=params["regThreshold"]
        self.subThreshold=params["subThreshold"]
        self.winThreshold=params["winThreshold"]
