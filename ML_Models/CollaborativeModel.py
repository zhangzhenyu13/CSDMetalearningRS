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


if __name__ == '__main__':
    data = DataSetTopcoderCluster()
    # regression
    data.CommitRegressionData()

    model = CFRegressor(linear_model.LinearRegression(),data.n_clusters)
    model.name = "linearRegressorCommit"
    model.dataSet = data
    for k in range(data.n_clusters):
        data.loadClusters(k)
        model.setLocality(k)
        model.trainModel()

        Y_predict1 = model.predict(data.testX)
        print("cluster:",k,", test mse=%f" % (metrics.mean_squared_error(data.testLabel, Y_predict1)))
    model.saveModel()
    model.loadModel()
    
