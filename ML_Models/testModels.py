from ML_Models.CascadingModel import *
from ML_Models.ModelTuning import bestPDIG
from DataPrepare.TopcoderDataSet import TopcoderWin

#cascading models
def testCascadingModel(tasktype):
    data=TopcoderWin(tasktype,testratio=1,validateratio=0)
    data.setParameter(tasktype,2,True)
    data.loadData()
    data.WinClassificationData()

    taskids=data.taskids[:data.testPoint]
    mymetric=model.mymetric
    mymetric.callall=False
    print("\n meta-learning model top k acc")
    Y_predict2=model.predict(data.testX,taskids)
    for k in (1,2,3,4,5):
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
    #exit(10)
    print()
    mymetric.verbose=0
    print("tuning cascading models")
    for k in(1,3,5):
        model.topK=k
        model.searchParameters(data)
        model.saveConf()

    mymetric.verbose=1
    model.setVerbose(1)

    print("\n meta-learning model top k acc")
    for k in (1,3,5):
        model.topK=k
        model.loadConf()

        Y_predict2=model.predict(data.testX,taskids)
        #metrics
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

    #mymetric.verbose=0
    #params=bestPDIG(mymetric,Y_predict2,data)
    #print(params)

    #exit(10)
if __name__ == '__main__':

    '''
    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }
    '''

    tasktype="Content Creation"
    model=CascadingModel(tasktype)
    model.loadModel()
    model.setVerbose(1)
    testCascadingModel(tasktype)
