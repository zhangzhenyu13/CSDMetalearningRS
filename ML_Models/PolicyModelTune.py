from ML_Models.UserMetrics import *
import numpy as np
from Utility.TagsDef import getUsers
import json
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.DNNModel import DNNCLassifier
from ML_Models.EnsembleModel import EnsembleClassifier

class PolicyModel:
    def initData(self):

        self.mymetric=TopKMetrics(tasktype=self.datatype,testMode=True)
        self.subExpr=self.mymetric.subRank
        self.userIndex=getUsers(self.datatype+"-test",mode=2)

        print("model init for %d users"%len(self.userIndex),"%d tasks"%len(self.subExpr))

    def __init__(self,tasktype=None,datatype=None):
        #meta-learners
        self.availableModels={
            0:EnsembleClassifier,
            1:XGBoostClassifier,
            2:DNNCLassifier
        }
        self.metaReg=1
        self.metaSub=1
        self.metaWin=1
        #parameters

        #aux info
        self.tasktype=tasktype
        self.datatype=self.tasktype
        if datatype is not None:
            self.datatype=datatype
        self.initData()
        self.name=tasktype+"rulePredictor"

    def testResults(self,trueLabel,topReg,topSub,topWin,top_R,top_S):
        '''

        :param topReg: 2 dim (taskNum,userNum)
        :param topSub: 2 dim (taskNum,userNum)
        :param topWin: 2 dim (taskNum,userNum)
        :param top_R:  0~1
        :param top_S: 0~1
        :return: predicts
        '''
        taskNum=len(topReg)
        userNum=len(self.userIndex)

        trueLabel=np.reshape(trueLabel,newshape=(taskNum,userNum))

        topRN=int(top_R*len(self.userIndex))
        topSN=int(top_S*len(self.userIndex))


        mrr=np.zeros(shape=taskNum,dtype=np.float32)
        acc3=np.zeros(shape=taskNum,dtype=np.int)
        acc5=np.zeros(shape=taskNum,dtype=np.int)
        acc10=np.zeros(shape=taskNum,dtype=np.int)


        for i in range(taskNum):
            trueY=trueLabel[i]
            trueY=np.where(trueY==1)[0]
            if len(trueY)==0:continue
            trueY=set(trueY)

            predictY=copy.deepcopy(topWin[i])


            topRY=topReg[i,:topRN]
            topSY=topSub[i,:topSN]

            for pos in range(userNum):
                if pos not  in topRY:
                    predictY[pos]=0
                    continue
                if pos not in topSY:
                    predictY[pos]=0

            predictY,_=self.mymetric.getTopKonPossibility(predictY,100000)

            com=trueY.intersection(predictY[:3])
            if len(com)==0:
                acc3[i]=0
            else:
                acc3[i]=1
            com=trueY.intersection(predictY[:5])
            if len(com)==0:
                acc5[i]=0
            else:
                acc5[i]=1
                com=trueY.intersection(predictY[:10])
            if len(com)==0:
                acc10[i]=0
            else:
                acc10[i]=1
            #mrr
            for j in range(len(predictY)):
                if predictY[j] in trueY:
                    mrr[i]=1.0/(1.0+j)


                break

        MRR=np.mean(mrr)
        ACC3=np.mean(acc3)
        ACC5=np.mean(acc5)
        ACC10=np.mean(acc10)

        #print(self.tasktype,"top 3 5 and 10 acc=",ACC3,ACC5,ACC10,"mrr=",MRR)

        return ACC3,ACC5,ACC10,MRR

    def TuneTempResults(self,X):
        regModels=[]
        subModels=[]
        winModels=[]

        tasktype=self.tasktype

        for i in self.availableModels.keys():
            regModel=self.availableModels[i]()
            subModel=self.availableModels[i]()
            winModel=self.availableModels[i]()
            winModel.name=tasktype+"-classifierWin"
            if  "#" in tasktype:
                pos=tasktype.find("#")
                regModel.name=tasktype[:pos]+"-classifierReg"
                subModel.name=tasktype[:pos]+"-classifierSub"
            else:
                regModel.name=tasktype+"-classifierReg"
                subModel.name=tasktype+"-classifierSub"

            regModel.loadModel()
            subModel.loadModel()
            winModel.loadModel()
            regModels.append(regModel)
            subModels.append(subModel)
            winModels.append(winModel)

        print("all meta models are loaded")


        regYs=[]
        subYs=[]
        winYs=[]
        taskNum=len(X)//len(self.userIndex)
        userNum=len(self.userIndex)

        for i in self.availableModels.keys():

            regModel,subModel,winModel=regModels[i],subModels[i],winModels[i]
            regY=regModel.predict(X)
            subY=subModel.predict(X)
            winY=winModel.predict(X)
            regY=np.reshape(regY,newshape=(taskNum,userNum))
            subY=np.reshape(subY,newshape=(taskNum,userNum))
            winY=np.reshape(winY,newshape=(taskNum,userNum))


            regYs.append(regY)
            subYs.append(subY)
            winYs.append(winY)

        for i in range(len(regYs)):


            regY,subY=regYs[i],subYs[i]

            #print(regY.shape)
            for j in range(taskNum):

                topReg,_=self.mymetric.getTopKonPossibility(regY[j],10000)
                topSub,_=self.mymetric.getTopKonPossibility(subY[j],10000)

                regY[j]=topReg
                subY[j]=topSub

            regYs[i],subYs[i]=regY,subY

        print("all temp results are generated")

        return regYs,subYs,winYs


def generateSearchData(saveData=True):
    print("\n begin Search\n")

    featureData=[]
    bestfeature={
            "a1":0,
            "a2":0,
            "a3":0,
            "regt":0.1,
            "subt":0.1,
        }

    curfeature=copy.deepcopy(bestfeature)
    bestacc3=0
    bestacc5=0
    bestacc10=0

    bestmrr=0
    featureAll=[]

    for a1 in featureSpace["a1"]:
        for a2 in featureSpace["a2"]:
            for a3 in featureSpace["a3"]:
                for top_r in featureSpace["regt"]:
                    for top_s in featureSpace["subt"]:
                        featureAll.append((a1,a2,a3,top_r,top_s))
    print("searching space size=%d"%len(featureAll))

    count=0
    for f in featureAll:
        a1,a2,a3,top_r,top_s=f

        regY=regYs[a1]
        subY=subYs[a2]
        winY=winYs[a3]
        acc3,acc5,acc10,mrr=model.testResults(data.testLabel,regY,subY,winY,top_r,top_s)

        featuredata=list(f)+[acc3,acc5,acc10,mrr]
        featureData.append(copy.deepcopy(featuredata))

        if acc3>bestacc3:
            bestacc3=acc3
        if acc5>bestacc5:
            bestacc5=acc5
        if acc10>bestacc10:
            bestacc10=acc10
        if mrr>bestmrr:
            bestmrr=mrr
        count+=1
        if count%100==0:
            print("step#%d best acc for top 3 5 and 10=(%f,%f,%f),and mrr =%f"%(count,bestacc3,bestacc5,bestacc10,bestmrr))

    print("feature data size",len(featureData))
    print("\nbest acc for top 3 5 and 10=(%f,%f,%f),and mrr =%f\n"%(bestacc3,bestacc5,bestacc10,bestmrr))

    if saveData:
        print("save Meta Data for",tasktype)
        with open("../data/MetaData/"+tasktype+".pkl","wb") as f:
            pickle.dump(featureData,f)

    return featureData

if __name__ == '__main__':
    from DataPrepare.TopcoderDataSet import TopcoderWin
    from Utility import SelectedTaskTypes
    import multiprocessing

    mode=2

    featureSpace={
                "a1":[0,1,2],
                "a2":[0,1,2],
                "a3":[0,1,2],
                "regt":[i/10.0 for i in range(1,11)],
                "subt":[i/10.0 for i in range(1,11)],
            }
    featureSize=len(featureSpace["a1"])*len(featureSpace["a2"])*len(featureSpace["a3"])*\
        len(featureSpace["regt"])*len(featureSpace["subt"])

    print(featureSpace)


    modelType="Assembly Competition#0"
    dataType="Code#2"
    data=TopcoderWin(dataType,testratio=1,validateratio=0)
    data.setParameter(dataType,2,True)
    data.loadData()
    data.WinClassificationData()

    tasktypes=SelectedTaskTypes.loadTaskTypes()
    tasktypes=list(tasktypes["keeped"])
    tasktypes.sort()

    # search
    model=PolicyModel(modelType,dataType)
    regYs,subYs,winYs=model.TuneTempResults(data.testX)
    generateSearchData(False)
