from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers
import json
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.DNNModel import DNNCLassifier
from ML_Models.EnsembleModel import EnsembleClassifier
from DataPrepare.DataContainer import UserHistoryGenerator

class CascadingModel:
    userHG=UserHistoryGenerator(True)
    def __init__(self):
        self.regModel=None
        self.subModel=None
        self.winModel=None
        self.regThreshold=0
        self.subThreshold=0
        self.winhn=0
        self.subhn=0
        self.topSN=1
        self.availableModels={
            1:DNNCLassifier,
            2:EnsembleClassifier,
            3:XGBoostClassifier
        }
        self.selKeys=[1,1,1]
        self.verbose=1
    def setVerbose(self,verbose):
        self.verbose=verbose
        self.regModel.verbose=self.verbose-1
        self.subModel.verbose=self.verbose-1
        self.winModel.verbose=self.verbose-1

    def loadModel(self,tasktype):
        #print(self.selKeys)
        models=[
            self.availableModels[self.selKeys[0]],
            self.availableModels[self.selKeys[1]],
            self.availableModels[self.selKeys[2]]
        ]
        print(len(models),models)

        self.regModel=models[0]()
        self.subModel=models[1]()
        self.winModel=models[2]()

        self.winModel.name=tasktype+"-classifierWin"

        if  "#" in tasktype:
            pos=tasktype.find("#")
            self.regModel.name=tasktype[:pos]+"-classifierReg"
            self.subModel.name=tasktype[:pos]+"-classifierSub"
        else:
            self.regModel.name=tasktype+"-classifierReg"
            self.subModel.name=tasktype+"-classifierSub"

        self.regModel.loadModel()
        self.subModel.loadModel()
        self.winModel.loadModel()


        self.name=tasktype+"-classifierRules"

        self.mymetric=TopKMetrics(tasktype=tasktype,testMode=True)
        self.subExpr=self.mymetric.subRank
        self.scoreExpr=self.mymetric.scoreRank
        self.userIndex=getUsers(tasktype+"-test",mode=2)
        userdata=CascadingModel.userHG.loadActiveUserHistory(tasktype,2)
        self.userdata={}
        for name in self.userIndex:
            self.userdata[name]={"win":0,"sub":0}
            subtasks=userdata[name]["subtasks"]
            ranks=subtasks[4]
            subnums=subtasks[1]
            self.userdata[name]["win"]=len(np.where(ranks==0)[0])
            self.userdata[name]["sub"]=np.sum(subnums)

        self.threshold=self.winModel.threshold

        try:
            self.loadConf()
        except:
            print("meta models config loading failed")

    def predict(self,X,taskids):
        if self.verbose>0:
            print("Cascading Model is predicting for %d users"%len(self.userIndex))
        regY=self.regModel.predict(X)
        subY=self.subModel.predict(X)
        winY=self.winModel.predict(X)

        Y=np.zeros(shape=len(X))
        taskNum=len(X)//len(self.userIndex)

        for i in range(taskNum):
            for j in range(len(self.userIndex)):
                pos=i*len(self.userIndex)+j
                taskid=taskids[pos]
                #init
                if self.userdata[self.userIndex[j]]["win"]<self.winhn:
                    continue
                if self.userdata[self.userIndex[j]]["sub"]<self.subhn:
                    continue
                #reg
                if regY[pos]<self.regThreshold:
                    continue

                #sub
                topN=int(self.topSN*len(self.userIndex))
                selectedusers,_ =self.mymetric.getTopKonDIGRank(self.subExpr[taskid]["ranks"],topN)

                if subY[pos]<self.subThreshold and j not in selectedusers:
                    continue
                #winner

                Y[pos]=winY[pos]

        return Y

    def saveConf(self):
        params={"regThreshold":self.regThreshold,
                "subThreshold":self.subThreshold,
                "topSN":self.topSN,
                "selKeys":self.selKeys,
                "winhn":self.winhn,"subhn":self.subhn}

        with open("../data/saved_ML_models/MetaPredictor/"+self.name+".json","w") as f:
            json.dump(params,f)

    def loadConf(self):
        with open("../data/saved_ML_models/MetaPredictor/"+self.name+".json","w") as f:
            params=json.load(f)
        self.regThreshold=params["regThreshold"]
        self.subThreshold=params["subThreshold"]
        self.topSN=params["topSN"]
        self.selKeys=params["selKeys"]
        self.winhn,self.subhn=params["winhn"],params["subhn"]
