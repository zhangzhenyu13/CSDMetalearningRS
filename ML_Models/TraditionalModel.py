from ML_Models.Model_def import *
from sklearn import svm,linear_model
from sklearn import ensemble
from sklearn import metrics
import time

#define set of regressor
regressor={
    "svr":svm.SVR()
}
#define set of classifier
classifier={
    "svc":svm.SVC()
}
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
    def __init__(self,classifier):
        ML_model.__init__(self)
        self.model=classifier
    def predict(self,X):
        print(self.name,"is predicting")
        Y=self.model.predict(X)
        return Y
    def trainModel(self):
        print("training")
        t0=time.time()
        self.model.fit(self.dataSet.trainX,self.dataSet.trainLabel)
        t1=time.time()
        score=metrics.accuracy_score(self.dataSet.trainLabel,self.model.predict(self.dataSet.trainX))
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"train score=%f"%score)

#test the performance
if __name__ == '__main__':

    data=DataSetTopcoder()
    #regression
    data.CommitRegressionData()
    model=TraditionalRegressor(svm.NuSVR())
    model.name="linearRegressorCommit"
    model.dataSet=data
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict1=model.predict(data.testX)
    print("test mse=%f"%(metrics.mean_squared_error(data.testLabel,Y_predict1)))
    print("navie commit probability=",np.sum(data.testLabel>0)/len(data.testLabel),"mean num",np.mean(data.testLabel))
    #classification
    data.WinClassificationData()
    model=TraditionalClassifier(ensemble.RandomForestClassifier())
    model.dataSet=data
    model.name="linearClassifierCommit"
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2,normalize=False)))

