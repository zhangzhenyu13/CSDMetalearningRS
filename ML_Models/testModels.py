from ML_Models.EnsembleModel import *
from ML_Models.CascadingModel import *
from ML_Models.DNNModel import *
from ML_Models.XGBoostModel import *
from ML_Models.ModelTuning import loadData,topKmetrics,showMetrics

#run cascading models
def testCascadingModel(tasktype,metamodels):
    data=loadData(tasktype,2)

    model=CascadingModel()
    model.loadModel(tasktype,metamodels)
    taskids=data.taskids[:data.testPoint]
    Y_predict2=model.predict(data.testX,taskids)

    showMetrics(Y_predict2,data,model.threshold)

    topKmetrics(Y_predict2,data)




if __name__ == '__main__':

    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }


    selectedmodel=3

    tasktype="Architecture"

    ml_models=[ml_model[1],ml_model[1],ml_model[1]]

    testCascadingModel(tasktype,ml_models)

