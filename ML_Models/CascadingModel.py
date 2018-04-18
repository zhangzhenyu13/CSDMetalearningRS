from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers
import json
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.DNNModel import DNNCLassifier
from ML_Models.EnsembleModel import EnsembleClassifier

class CascadingModel:
    def initData(self):
        self.mymetric=TopKMetrics(tasktype=self.tasktype,testMode=True)
        self.subExpr=self.mymetric.subRank
        self.userIndex=getUsers(self.tasktype+"-test",mode=2)
        print("model init for %d users"%len(self.userIndex),"%d tasks"%len(self.subExpr))

    def __init__(self,tasktype):
        #meta-learners
        self.regModel=None
        self.subModel=None
        self.winModel=None
        self.availableModels={
            1:EnsembleClassifier,
            2:XGBoostClassifier,
            3:DNNCLassifier
        }
        self.metaReg=1
        self.metaSub=1
        self.metaWin=1
        #parameters
        self.regThreshold=0
        self.subThreshold=0
        self.topDig=1

        #aux info
        self.verbose=1
        self.tasktype=tasktype
        self.initData()
        self.topK=-1
        self.name=tasktype+"rulePredictor"

    def setVerbose(self,verbose):
        self.verbose=verbose
        self.regModel.verbose=self.verbose-1
        self.subModel.verbose=self.verbose-1
        self.winModel.verbose=self.verbose-1

    def loadModel(self):
        tasktype=self.tasktype

        self.regModel=self.availableModels[self.metaReg]()
        self.subModel=self.availableModels[self.metaSub]()
        self.winModel=self.availableModels[self.metaWin]()

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
        if self.verbose>0:
            print("meta learner loaded",self.regModel,self.subModel,self.winModel)

    def searchParameters(self,data):


        print("search best parameters for top%d prediction"%self.topK)
        minRT=self.regThreshold
        minST=self.subThreshold
        minDn=self.topDig

        reg=1
        sub=1
        win=1
        maxAcc=0
        processing=0
        for self.metaReg in (1,2,3):
            for self.metaSub in (1,2,3):
                for self.metaWin in (1,2,3):
                    self.setVerbose(1)
                    self.loadModel()
                    processing+=1
                    print("process=%d/27,maxAcc=%f"%(processing,maxAcc))
                    self.setVerbose(0)

                    for self.regThreshold in range(0,10):
                        self.regThreshold=self.regThreshold/10

                        for self.subThreshold in range(0,10):
                            self.subThreshold=self.subThreshold/10

                            for self.topDig in range(0,11):
                                self.topDig=self.topDig/10

                                Y=self.predict(data.testX,data.taskids)
                                Y=self.mymetric.topKPossibleUsers(Y,data,self.topK)
                                acc=np.mean(Y)
                                print("top%d"%self.topK,acc,self.regThreshold,self.subThreshold,self.topDig)
                                if maxAcc<acc:
                                    #update para record when acc is higher
                                    maxAcc=acc

                                    minRT=self.regThreshold
                                    minST=self.subThreshold
                                    minDn=self.topDig

                                    reg=self.metaReg
                                    sub=self.metaSub
                                    win=self.metaWin

        self.regThreshold,self.subThreshold,self.topDig=minRT,minST,minDn

        self.metaReg,self.metaSub,self.metaWin=reg,sub,win
        print("\n searched best parameters for top%d(acc=%f) with meta learners="%(self.topK,maxAcc),
              self.metaReg,self.metaSub,self.metaWin)

        print("regThreshold=%f,subThreshold=%f,topDig=%f\n"%(
            self.regThreshold,self.subThreshold,self.topDig
        ))

    def predict(self,X,taskids):

        if self.verbose>0:
            print("Cascading Model is predicting top %d for %d users,parameters are"%(self.topK,
                    len(self.userIndex)))
            print("regThreshold=%f, subThreshold=%f, DigThreshold=%f"%(self.regThreshold,self.subThreshold,
                  self.topDig))
            print()

        regY=self.regModel.predict(X)
        subY=self.subModel.predict(X)
        winY=self.winModel.predict(X)

        Y=np.zeros(shape=len(X))
        taskNum=len(X)//len(self.userIndex)
        for i in range(taskNum):
            for j in range(len(self.userIndex)):
                pos=i*len(self.userIndex)+j
                taskid=taskids[pos]
                #reg

                if regY[pos]<self.regThreshold:
                    #count+=1
                    continue

                #sub
                topN=int(self.topDig*len(self.userIndex))
                selectedusers,_ =self.mymetric.getTopKonDIGRank(self.subExpr[taskid]["ranks"],topN)
                #print(taskid,len(selectedusers),len(self.subExpr[taskid]["ranks"]),topN)
                if subY[pos]<self.subThreshold or j not in selectedusers:
                    #count+=1
                    continue
                #winner

                Y[pos]=winY[pos]
        #print("filtered %d in predict"%count)
        return Y

    def saveConf(self):
        params={"regThreshold":self.regThreshold,
                "subThreshold":self.subThreshold,
                "topDig":self.topDig,
                "metaReg":self.metaReg,"metaSub":self.metaSub,"metaWin":self.metaWin
                }

        with open("../data/saved_ML_models/MetaPredictor/"+self.name+"-top"+str(self.topK)+".json","w") as f:
            json.dump(params,f)

    def loadConf(self):
        with open("../data/saved_ML_models/MetaPredictor/"+self.name+"-top"+str(self.topK)+".json","r") as f:
                params=json.load(f)

        self.regThreshold=params["regThreshold"]
        self.subThreshold=params["subThreshold"]
        self.topDig=params["topDig"]
        self.metaReg=params["metaReg"]
        self.metaSub=params["metaSub"]
        self.metaWin=params["metaWin"]
