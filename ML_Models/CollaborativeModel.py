from ML_Models.Model_def import *
from sklearn import svm,linear_model
from sklearn import ensemble
from sklearn import metrics
import time

class CFRegressor(ML_model):
    def __init__(self,n_neighborhood,regressor):
        ML_model.__init__(self)

        self.model={}
        self.k_no=None
        for  k in range(n_neighborhood):
            self.model[k]=regressor()

    def setLocality(self,k_no):
        self.k_no=k_no
    def predict(self, X):
        print(self.name, "is predicting")
        Y = self.model[self.k_no].predict(X)
        return Y

    def trainModel(self):
        print("training")
        t0 = time.time()
        self.model[self.k_no].fit(self.dataSet.trainX, self.dataSet.trainLabel)
        t1 = time.time()
        mse = metrics.mean_squared_error(self.dataSet.trainLabel, self.model[self.k_no].predict(self.dataSet.trainX))
        print("model", self.name, "training finished in %ds" % (t1 - t0), "train mse=%f" % mse)

class CFClassifier(ML_model):
    def __init__(self,n_neighborhood,regressor):
        ML_model.__init__(self)

        self.model={}
        self.k_no=None
        for  k in range(n_neighborhood):
            self.model[k]=regressor()
    def setLocality(self,k_no):
        self.k_no=k_no
    def predict(self, X):
        print(self.name, "is predicting")
        Y = self.model[self.k_no].predict(X)
        return Y

    def trainModel(self):
        print("training")
        t0 = time.time()
        self.model[self.k_no].fit(self.dataSet.trainX, self.dataSet.trainLabel)
        t1 = time.time()
        accuracy = metrics.accuracy_score(self.dataSet.trainLabel, self.model[self.k_no].predict(self.dataSet.trainX))
        print("model", self.name, "training finished in %ds" % (t1 - t0), "train mse=%f" % accuracy)

if __name__ == '__main__':
    data = DataSetTopcoderCluster()

    model = CFClassifier(data.n_clusters,ensemble.RandomForestClassifier)
    model.name = "classifier_cluster"
    model.dataSet = data
    for k in range(data.n_clusters):
        data.loadClusters(k)
        # regression
        data.CommitClassificationData()
        model.setLocality(k)
        model.trainModel()

        print("test size=",len(data.testLabel))
        if len(data.testLabel)<2:
            print("no data to predict")
            print()
            continue
        Y_predict1 = model.predict(data.testX)
        print("cluster:",k,", test accuracy=%f" % (metrics.accuracy_score(data.testLabel, Y_predict1)))
        print()

    model.saveModel()
    model.loadModel()

