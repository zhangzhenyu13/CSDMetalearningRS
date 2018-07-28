from DataPrepare.TopcoderDataSet import *
from sklearn import metrics

DataMode={
    0:TopcoderReg,
    1:TopcoderSub,
    2:TopcoderWin
}

def loadData(tasktype,mode):
    data=DataMode[mode](tasktype,testratio=0.2,validateratio=0.1)
    data.loadData()
    if mode==0:
        data.RegisterClassificationData()
    if mode==1:
        data.SubmitClassificationData()
    if mode==2:
        data.WinClassificationData()

    data.trainX,data.trainLabel=data.ReSampling(data.trainX,data.trainLabel)
    data.validateX,data.validateLabel=data.ReSampling(data.validateX,data.validateLabel)

    return data

def showMetrics(Y_predict2,data,threshold):
    Y_predict1=np.array(Y_predict2>threshold,dtype=np.int)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict1)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict1))


def topKmetrics(mymetric,Y_predict2,data):
    print("\n meta-learning model top k acc")
    for k in (1,3,5):
        acc=mymetric.topKPossibleUsers(Y_predict2,data.testLabel,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        '''
        acc=mymetric.topKDIGUsers(data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)
        '''


        print()

