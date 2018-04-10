from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import svm,linear_model,naive_bayes,tree
from sklearn import ensemble
from sklearn import metrics
import time


class TraditionalClassifier(ML_model):
    def ModelTuning(self):
        candite_selection = {
            "RandomFrorest": ensemble.RandomForestClassifier(),
            "ExtraForest": ensemble.ExtraTreesClassifier(),
            #"AdaBoost": ensemble.AdaBoostClassifier(),
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
    def trainModel(self,dataSet):
        print("training")
        t0=time.time()
        candidate_model=self.ModelTuning()
        max_acc=0
        sel_model=None
        dataSet.validateX,dataSet.validateLabel=dataSet.ReSampling(
            dataSet.validateX,dataSet.validateLabel
        )
        for key in candidate_model.keys():
            self.model=candidate_model[key]
            if self.model is not None:
                self.model.fit(dataSet.trainX,dataSet.trainLabel)
                v_predict=self.model.predict(dataSet.validateX)

                acc=metrics.accuracy_score(dataSet.validateLabel,v_predict)
                print(key,acc)
                if acc>max_acc:
                    sel_model=self.model
                    max_acc=acc
        self.model=sel_model

        trainData=np.concatenate((dataSet.trainX,dataSet.validateX),axis=0)
        trainLabel=np.concatenate((dataSet.trainLabel,dataSet.validateLabel),axis=0)
        self.model.fit(trainData,trainLabel)

        t1=time.time()
        score=metrics.accuracy_score(dataSet.validateLabel,self.model.predict(dataSet.validateX))
        cm=metrics.confusion_matrix(dataSet.validateLabel,self.model.predict(dataSet.validateX))
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)





