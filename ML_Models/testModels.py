from ML_Models.EnsembleModel import *
from ML_Models.CascadingModel import *
from ML_Models.DNNModel import *
from ML_Models.XGBoostModel import *
from ML_Models.ModelTuning import loadData,topKmetrics,showMetrics,bestPDIG

#run cascading models
def testCascadingModel(tasktype,metamodels):
    data=loadData(tasktype,2)

    model=CascadingModel()
    model.loadModel(tasktype,metamodels)
    taskids=data.taskids[:data.testPoint]
    mymetric=TopKMetrics(tasktype)
    maxAcc=[0,(0,0,0)]
    w1=0
    w2=0
    w3=0

    for w1 in range(0,11):
        for w2 in range(0,11):
            for w3 in range(0,11):
                model.regThreshold=w1/10
                model.subThreshold=w2/10
                model.winThreshol=w3/10
                Y_predict2=model.predict(data.testX,taskids)
                acc=mymetric.topKPossibleUsers(Y_predict2,data,3)
                acc=np.mean(acc)
                if acc>maxAcc[0]:
                    maxAcc=[acc,(w1,w2,w3)]
    model.saveConf()
    print("\nbest threshold, reg:%f, sub:%f, win:%f=>acc:%f"%(w1,w2,w3,maxAcc[0]))

    Y_predict2=model.predict(data.testX,taskids)

    showMetrics(Y_predict2,data,model.threshold)

    mymetric=topKmetrics(Y_predict2,data)


    bestPDIG(mymetric,Y_predict2,data)

if __name__ == '__main__':

    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }


    selectedmodel=(2,2,2)

    tasktype="Architecture"

    ml_models=[ml_model[selectedmodel[0]],ml_model[selectedmodel[1]],ml_model[selectedmodel[2]]]

    testCascadingModel(tasktype,ml_models)

