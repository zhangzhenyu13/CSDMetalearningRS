from ML_Models.Model_def import *
from sklearn import svm,linear_model,neighbors,naive_bayes
from sklearn import ensemble
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
        candite_selection={
            "RandomFrorest":ensemble.RandomForestClassifier(),
            "ExtraForest":ensemble.ExtraTreesClassifier(),
            "AdaBoost":ensemble.AdaBoostClassifier(),
            "GradientBoost":ensemble.GradientBoostingClassifier(),
            "Bagging":ensemble.BaggingClassifier(),
            "SVM":svm.SVC()
        }
        return candite_selection
    def __init__(self,neighborhoods,name):
        ML_model.__init__(self)
        self.name=name
        self.assessModel=classificationAssess(self.name)
        self.model={}
        self.neighborhood=None
        for  k in neighborhoods:
            self.model[k]=None
    def setLocality(self,neighborhood):
        self.neighborhood=neighborhood
    def predict(self, X):
        print(self.name, "is predicting")
        Y = self.model[self.neighborhood].predict(X)
        return Y
    def evaluate(self,tri_model):
        Y=tri_model.predict(self.dataSet.validateX)
        acc=metrics.accuracy_score(self.dataSet.validateLabel,Y)
        cfm=metrics.confusion_matrix(self.dataSet.validateLabel,Y)
        recall,precision=self.assessModel.processCFM(cfm=cfm,acc=acc)
        return acc,recall, precision
    def trainModel(self):
        print("training")
        t0 = time.time()
        max_acc=0
        tri_model=None
        tuned_model=None
        model_selection=self.ModelTuning()
        for cls in model_selection.keys():
            tri_model=model_selection[cls]
            tri_model.fit(self.dataSet.trainX, self.dataSet.trainLabel)
            accuracy,recall,precision=self.evaluate(tri_model=tri_model)
            if accuracy>max_acc:
                max_acc=accuracy
                tuned_model=tri_model
            print(cls,"validation accuracy=%f, recall=%f, precision=%f"%(accuracy,recall,precision))

        self.model[self.neighborhood]=tuned_model
        t1=time.time()
        print("model", self.name, "training finished in %ds with %d instances" % (t1 - t0,len(self.dataSet.trainLabel)),
              "tuning accuracy=%f" % max_acc)
        return max_acc
if __name__ == '__main__':

    data = DataSetTopcoderCluster(splitraio=0.7)

    model = CFClassifier(data.clusternames,"local_classifier_Sub")
    model.dataSet = data
    y=[]
    count=0
    sum=0
    sum2=0

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

    plt.plot(np.arange(len(y)),y,marker='o')
    plt.show()
    model.saveModel()
    model.loadModel()
    print("average accuracy=",sum2/len(data.clusternames),"weighted average accuracy=",sum/count)
