from ML_Models.DNNModel import DNNCLassifier
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.EnsembleModel import EnsembleClassifier
from ML_Models.BaselineModel import *
import numpy as np

def testAcc(mymetric,model,data,testK=(3,5,10)):
    #mymetric.verbose=2
    Y_predict2=model.predict(data.testX)
    print("\n meta-learning model top k acc")
    for k in testK:
        acc=mymetric.topKPossibleUsers(Y_predict2,data.testLabel,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d acc"%k,acc)

        '''
        acc=mymetric.topKRUsers(Y_predict2,data.testLabel,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,data.testLabel,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)
        '''


        print()
    #exit(10)
def testMRR(mymetric,model,data):
    Y_predict2=model.predict(data.testX)
    print("\n meta-learning model mrr metrics")

    mrr=mymetric.getAllMRR(Y_predict2,data.testLabel)
    mrr=np.mean(mrr)
    print(data.tasktype,"mrr=",mrr)
    print()

def transferLearningTest(tasktypes,datatype):
    data=TopcoderWin(datatype,testratio=1,validateratio=0)
    data.setParameter(datatype,2,True)
    data.loadData()
    data.WinClassificationData()
    mymetric=TopKMetrics(tasktype=datatype,testMode=True)

    for tasktype in tasktypes["keeped"]:

        model=DNNCLassifier()
        model.name=tasktype+"-classifier"+ModeTag[mode]
        model.loadModel()

        testAcc(mymetric=mymetric,model=model,data=data,testK=(1,2))

    exit(10)

if __name__ == '__main__':
    from Utility.TagsDef import ModeTag
    from DataPrepare.TopcoderDataSet import TopcoderWin
    from Utility import SelectedTaskTypes
    from ML_Models.UserMetrics import TopKMetrics
    mode=2

    tasktypes=SelectedTaskTypes.loadTaskTypes()
    #transferLearningTest(tasktypes,"Design")
    tasktypes=list(tasktypes["clustered"])
    tasktypes.sort()
    for tasktype in tasktypes:
        #if tasktype!="Test Suites":continue
        #if "Code" in tasktype or "Assembly" in tasktype or "First2Finish" in tasktype:
        #    continue

        mymetric=TopKMetrics(tasktype=tasktype,testMode=True)
        mymetric.callall=True
        model=XGBoostClassifier()
        model.name=tasktype+"-classifier"+ModeTag[mode]
        model.loadModel()

        data=TopcoderWin(tasktype,testratio=1,validateratio=0)
        data.setParameter(tasktype,2,True)
        data.loadData()
        data.WinClassificationData()

        testAcc(mymetric=mymetric,model=model,data=data,testK=(3,5,10))
        testMRR(mymetric=mymetric,model=model,data=data)
        print()
