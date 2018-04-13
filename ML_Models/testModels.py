from ML_Models.EnsembleModel import *
from ML_Models.CascadingModel import *
from ML_Models.DNNModel import *
from ML_Models.XGBoostModel import *
from DataPrepare.TopcoderDataSet import *
from sklearn import metrics
from ML_Models.XGBTuning import loadData

#run cascading models
def testCascadingModel(tasktype,metamodels):
    data=loadData(tasktype,2)

    model=CascadingModel()
    model.loadModel(tasktype,metamodels)
    taskids=data.taskids[:data.testPoint]
    Y_predict2=model.predict(data.testX,taskids)

    Y_predict1=np.array(Y_predict2>model.threshold,dtype=np.int)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict1)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict1))

    kacc=[data.tasktype]
    for k in (1,3,5,10):
        acc=topKPossibleUsers(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)
        kacc=kacc+[acc]
    print()


if __name__ == '__main__':

    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }


    selectedmodel=3

    tasktype="Architecture"

    ml_models=[ml_model[3],ml_model[3],ml_model[3]]

    testCascadingModel(tasktype,ml_models)
