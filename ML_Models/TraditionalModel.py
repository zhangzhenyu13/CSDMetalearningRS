from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import svm,linear_model,naive_bayes,tree
from sklearn import ensemble
from sklearn import metrics
import time
from imblearn import under_sampling,over_sampling
import matplotlib.pyplot as plt

#model container
class TraditionalRegressor(ML_model):
    def __init__(self,regressor):
        ML_model.__init__(self)
        self.model=regressor
    def predict(self,X):
        print(self.name,"is predicting")
        Y=self.model.predict(X)
        return Y
    def trainModel(self):
        print("training")
        t0=time.time()
        self.model.fit(self.dataSet.trainX,self.dataSet.trainLabel)
        t1=time.time()
        mse=metrics.mean_squared_error(self.dataSet.trainLabel,self.model.predict(self.dataSet.trainX))
        print("model",self.name,"training finished in %ds"%(t1-t0),"train mse=%f"%mse)

class TraditionalClassifier(ML_model):
    def ModelTuning(self):
        candite_selection = {
            "RandomFrorest": ensemble.RandomForestClassifier(),
            "ExtraForest": ensemble.ExtraTreesClassifier(),
            "AdaBoost": ensemble.AdaBoostClassifier(),
            #"GradientBoost": ensemble.GradientBoostingClassifier(),
            #"SVM": svm.SVC(C=0.9)
        }
        return candite_selection
    def __init__(self):
        ML_model.__init__(self)
    def predict(self,X):
        print(self.name,"is predicting")
        Y=self.model.predict(X)
        return Y
    def trainModel(self):
        print("training")
        t0=time.time()
        candidate_model=self.ModelTuning()
        max_acc=0
        sel_model=None
        for key in candidate_model.keys():
            self.model=candidate_model[key]
            if self.model is not None:
                self.model.fit(self.dataSet.trainX,self.dataSet.trainLabel)
                v_predict=self.model.predict(self.dataSet.validateX)

                acc=metrics.accuracy_score(self.dataSet.validateLabel,v_predict)
                print(key,acc)
                if acc>max_acc:
                    sel_model=self.model
                    max_acc=acc
        self.model=sel_model

        trainData=np.concatenate((data.trainX,data.validateX),axis=0)
        trainLabel=np.concatenate((data.trainLabel,data.validateLabel),axis=0)
        self.model.fit(trainData,trainLabel)

        t1=time.time()
        score=metrics.accuracy_score(self.dataSet.validateLabel,self.model.predict(self.dataSet.validateX))
        cm=metrics.confusion_matrix(self.dataSet.validateLabel,self.model.predict(self.dataSet.validateX))
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

def testClassification(data):
    model=TraditionalClassifier()
    model.dataSet=data
    model.name=data.tasktype+"-classifier(submission)"
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2,normalize=True)))
    print()
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict2))

    acc=topKAccuracy(Y_predict2,data,5)
    print(acc)
    print(np.mean(acc))

#test the performance
if __name__ == '__main__':

    with open("../data/TaskInstances/TaskIndex.data","rb") as f:
        tasktypes=pickle.load(f)
    mode=1
    for t in tasktypes:
        data=TopcoderWin(testratio=0.1,validateratio=0.1)
        data.setParameter(tasktype=t,mode=mode)
        data.loadData()
    

        #classification
        data.WinRankData()
        data.trainX,data.trainLabel=data.ReSampling(data.trainX,data.trainLabel,
                                                    over_sampling.ADASYN)

