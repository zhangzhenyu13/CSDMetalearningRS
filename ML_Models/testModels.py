from ML_Models.CascadingModel import *
from ML_Models.ModelTuning import bestPDIG
from DataPrepare.TopcoderDataSet import TopcoderWin
#cascading models

def searchParamForTopK(mymetric,taskids,data,k):
    print("search for top %d"%k)
    maxAcc=[0,(0,0,0,0,0)]
    for w1 in range(0,4):
        for w2 in range(0,4):
            for w3 in range(5,11):
                for w4 in range(1,6):
                    for w5 in range(1,6):
                        #params
                        model.regThreshold=w1/10
                        model.subThreshold=w2/10
                        model.topSN=w3/10
                        model.subhn=w4
                        model.winhn=w5
                        #predict test
                        Y_predict2=model.predict(data.testX,taskids)
                        acc=mymetric.topKPossibleUsers(Y_predict2,data,k)
                        acc=np.mean(acc)
                        if acc>maxAcc[0]:
                            maxAcc=[acc,(w1/10,w2/10,w3/10,w4,w5)]

    model.saveConf()
    w1,w2,w3,w4,w5=maxAcc[1]
    print("\nbest threshold, reg:%f, sub:%f, topSN:%f, subn:%d, winn:%d=> acc:%f"%(w1,w2,w3,w4,w5,maxAcc[0]))

    return maxAcc

def testCascadingModel(tasktype):
    data=TopcoderWin(tasktype,testratio=1,validateratio=0)
    data.setParameter(tasktype,2,True)
    data.loadData()
    data.WinClassificationData()

    taskids=data.taskids[:data.testPoint]
    mymetric=model.mymetric

    print("\n meta-learning model top k acc")
    Y_predict2=model.predict(data.testX,taskids)
    for k in (1,):
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
    exit(10)
    mymetric.verbose=0

    print("tuning cascading models")
    for k in(1,3,5):
        searchParamForTopK(mymetric,taskids,data,k)

    mymetric.verbose=1
    model.setVerbose(1)
    Y_predict2=model.predict(data.testX,taskids)

    print("\n meta-learning model top k acc")
    for k in (1,3,5,10):
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

    mymetric.verbose=0
    params=bestPDIG(mymetric,Y_predict2,data)
    print(params)

if __name__ == '__main__':

    '''
    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }
    '''

    for reg in (3,2,1):
        for sub in (3,2,1):
            for win in (3,2,1):
                tasktype="Content Creation"
                model=CascadingModel()
                model.selKeys=[reg,sub,win]
                model.loadModel(tasktype)
                model.setVerbose(0)
                testCascadingModel(tasktype)

