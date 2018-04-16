from ML_Models.DNNModel import DNNCLassifier
from ML_Models.XGBoostModel import XGBoostClassifier
from ML_Models.EnsembleModel import EnsembleClassifier
import numpy as np

def testAcc(mymetric,model,data):
    Y_predict2=model.predict(data.testX)
    print("\n meta-learning model top k acc")
    for k in (1,3,5):
        acc=mymetric.topKPossibleUsers(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKRUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        print()

if __name__ == '__main__':
    from Utility.TagsDef import ModeTag
    from DataPrepare.TopcoderDataSet import TopcoderWin
    from Utility import SelectedTaskTypes
    from ML_Models.UserMetrics import TopKMetrics
    mode=2
    tasktypes=SelectedTaskTypes.loadTaskTypes()

    for tasktype in tasktypes["keeped"]:

        mymetric=TopKMetrics(tasktype=tasktype,testMode=True)

        model=EnsembleClassifier()
        model.name=tasktype+"-classifier"+ModeTag[mode]
        model.loadModel()

        data=TopcoderWin(tasktype,testratio=1,validateratio=0)
        data.setParameter(tasktype,2,True)
        data.loadData()
        data.WinClassificationData()


        testAcc(mymetric=mymetric,model=model,data=data)
        print()
