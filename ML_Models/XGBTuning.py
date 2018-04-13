from ML_Models.CascadingModel import *
from ML_Models.XGBoostModel import *
from ML_Models.UserMetrics import *
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
def showMetrics(Y_predict2,threshold):
    Y_predict1=np.array(Y_predict2>threshold,dtype=np.int)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict1)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict1))

def testReg(data):
    model=XGBoostClassifier()
    model.name=data.tasktype+"-classifier(Sub)"
    model.trainModel(data)
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    showMetrics(Y_predict2,model.threshold)

def testSub(data):
    model=XGBoostClassifier()
    model.name=data.tasktype+"-classifier(Sub)"
    model.trainModel(data)
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    showMetrics(Y_predict2,model.threshold)


def testWin(data):
    model=XGBoostClassifier()
    model.name=data.tasktype+"-classifier(Win)"
    model.trainModel(data)
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    showMetrics(Y_predict2,model.threshold)


    for k in (1,3,5,10):
        acc=topKPossibleUsers(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=topKDIGUsers(data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

    print()

if __name__ == '__main__':
    tasktype="Architecture"
    mode=0
    data=loadData(tasktype,mode)
    testReg(data)
