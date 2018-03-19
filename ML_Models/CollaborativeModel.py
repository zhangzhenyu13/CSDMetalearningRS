from ML_Models.Model_def import *
from sklearn import svm,linear_model,neighbors
from sklearn import ensemble
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import pandas as pd
class CFRegressor(ML_model):
    def __init__(self,neighborhoods,regressor):
        ML_model.__init__(self)

        self.model={}
        self.neighborhood=None
        for  k in neighborhoods:
            self.model[k]=regressor()

    def setLocality(self,neighborhood):
        self.neighborhood=neighborhood
    def predict(self, X):
        print(self.name, "is predicting")
        Y = self.model[self.neighborhood].predict(X)
        return Y

    def trainModel(self):
        print("training")
        t0 = time.time()
        self.model[self.neighborhood].fit(self.dataSet.trainX, self.dataSet.trainLabel)
        t1 = time.time()
        mse = metrics.mean_squared_error(self.dataSet.trainLabel, self.model[self.neighborhood].predict(self.dataSet.trainX))
        print("model", self.name, "training finished in %ds" % (t1 - t0), "train mse=%f" % mse)

class CFClassifier(ML_model):
    def ModelTuning(self):
        self.selection={
            "UI Prototype Competition0.0":ensemble.RandomForestClassifier
        }
    def __init__(self,neighborhoods,classifier):
        ML_model.__init__(self)

        self.model={}
        self.neighborhood=None
        for  k in neighborhoods:
            self.model[k]=classifier()
    def setLocality(self,neighborhood):
        self.neighborhood=neighborhood
    def predict(self, X):
        print(self.name, "is predicting")
        Y = self.model[self.neighborhood].predict(X)
        return Y

    def trainModel(self):
        print("training")
        t0 = time.time()
        self.model[self.neighborhood].fit(self.dataSet.trainX, self.dataSet.trainLabel)
        t1 = time.time()
        accuracy = metrics.accuracy_score(self.dataSet.trainLabel, self.model[self.neighborhood].predict(self.dataSet.trainX))
        print("model", self.name, "training finished in %ds with %d instances" % (t1 - t0,len(self.dataSet.trainLabel)),
              "train accuracy=%f" % accuracy)
        return accuracy
if __name__ == '__main__':

    data = DataSetTopcoderCluster(splitraio=0.7)

    model = CFClassifier(data.clusternames,neighbors.KNeighborsClassifier)

    model.name = "classifier_cluster"
    model.dataSet = data
    y=[]
    count=0
    sum=0
    sum2=0

    assess=classificationAssess(model.name)
    for k in data.clusternames:
        data.loadClusters(k)
        # regression
        data.CommitClassificationData()
        model.setLocality(k)
        train_acc=model.trainModel()

        print("test size=",len(data.testLabel))
        if len(data.testLabel)<2:
            print("no data to predict")
            print()
            continue
        Y_predict1 = model.predict(data.testX)
        accuracy=metrics.accuracy_score(data.testLabel, Y_predict1)
        print("cluster:",k,", test accuracy=%f" % accuracy)
        cfm=metrics.confusion_matrix(data.testLabel,Y_predict1)
        print(cfm)
        print()

        y.append(accuracy)
        sum+=accuracy*len(data.testLabel)
        count+=len(data.testLabel)
        sum2+=accuracy
        record=(k,len(data.trainLabel),len(data.testLabel),train_acc,accuracy,cfm)
        assess.addValue(record)
    plt.plot(np.arange(len(y)),y,marker='o')
    assess.saveData()
    plt.show()
    model.saveModel()
    model.loadModel()
    print("average accuracy=",sum2/len(data.clusternames),"weighted average accuracy=",sum/count)
