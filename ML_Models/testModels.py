from ML_Models.EnsembleModel import *
from ML_Models.CascadingModel import *
from ML_Models.DNNModel import *
from ML_Models.XGBoostModel import *
from ML_Models.UserMetrics import topKAccuracyWithDIG,topKAccuracy,topKAccuracyOnSubset
from DataPrepare.TopcoderDataSet import *
from sklearn import metrics
import multiprocessing

def testRegClassification(tasktype,queue,model=EnsembleClassifier):
    data=TopcoderReg(tasktype,testratio=0.2,validateratio=0.1)

    data.loadData()

    data.RegisterClassificationData()
    data.trainX,data.trainLabel=data.ReSampling(data.trainX,data.trainLabel)
    data.validateX,data.validateLabel=data.ReSampling(data.validateX,data.validateLabel)

    model=model()
    model.name=data.tasktype+"-classifier(Reg)"
    model.trainModel(data)
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict2))
def testSubClassification(tasktype,queue,model=EnsembleClassifier):
    data=TopcoderSub(tasktype,testratio=0.2,validateratio=0.1)

    data.loadData()

    data.SubmitClassificationData()
    data.trainX,data.trainLabel=data.ReSampling(data.trainX,data.trainLabel)
    data.validateX,data.validateLabel=data.ReSampling(data.validateX,data.validateLabel)

    model=model()
    model.name=data.tasktype+"-classifier(Sub)"
    model.trainModel(data)
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict2))

def testWinClassification(tasktype,queue,model=EnsembleClassifier):
    data=TopcoderWin(tasktype,testratio=0.2,validateratio=0.1)
    data.loadData()
    data.WinClassificationData()
    data.trainX,data.trainLabel=data.ReSampling(data.trainX,data.trainLabel)
    data.validateX,data.validateLabel=data.ReSampling(data.validateX,data.validateLabel)

    model=model()
    model.name=data.tasktype+"-classifier(Win)"
    model.trainModel(data)
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict2))
    kacc=[data.tasktype]
    for k in (3,5,10,20):
        acc=topKAccuracyWithDIG(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)
        kacc=kacc+[acc]
    print()
    if queue is not None:
        queue.put(kacc)

def testCascadingModel(tasktype,queue,metamodels):
    data=TopcoderWin(tasktype,testratio=0.2,validateratio=0.1)
    data.loadData()
    data.WinClassificationData()

    model=CascadingModel()
    model.loadModel(tasktype,metamodels)
    taskids=data.taskids[:data.testPoint]
    Y_predict2=model.predict(data.testX,taskids)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict2))
    kacc=[data.tasktype]
    for k in (1,3,5,10):
        acc=topKAccuracyWithDIG(Y_predict2,data,k,True)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)
        kacc=kacc+[acc]
    print()
    if queue is not None:
        queue.put(kacc)

#test the performance
#parallel test method
def parallelRun():
    # begin test
    from Utility import SelectedTaskTypes
    tasktypes=SelectedTaskTypes.loadTaskTypes()

    queue=multiprocessing.Queue()
    pool_processes=[]
    for t in tasktypes["clustered"]:
        #if "First2Finish#3" not in t:
        #    continue
        #testWinRankClassification(t)

        p=multiprocessing.Process(target=testMethod[selectedmethod],args=(t,queue,ml_model[selectedmodel]))
        pool_processes.append(p)
        p.start()
        #p.join()
    for p in pool_processes:
        p.join()

    result=""
    while queue.empty()==False:
        data=queue.get()
        result=result+data[0]+" : %f"%data[1]
    with open("../data/runResults/testmodels"+str(selectedmethod)+".txt","w") as f:
        f.writelines(result)

if __name__ == '__main__':

    testMethod={
        1:testRegClassification,
        2:testSubClassification,
        3:testWinClassification,

    }
    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }

    selectedmethod=3

    selectedmodel=3

    tasktype="Code#0"
    testMethod[selectedmethod](tasktype,None,ml_model[selectedmodel])
    exit(10)

    ml_models=[]
    pos=0
    for i in range(1,1,2):
        ml_models.insert(pos,ml_model[i])
        pos+=1
    testCascadingModel(tasktype,None,ml_models)
