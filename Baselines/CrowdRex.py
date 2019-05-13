from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import tree,naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

class NBBayes(ML_model):

    def __init__(self):
        ML_model.__init__(self)

    def predict(self,X):
        if  len(X)==0:
            return np.array([],dtype=np.int)
        if self.verbose>0:
            print(self.name," NBBayes is predicting")

        Y=self.model.predict_proba(X)
        return Y[:,1]


    def trainModel(self,dataSet):
        print("training")
        t0=time.time()

        self.model=naive_bayes.GaussianNB()

        print("training label(2) test",Counter(dataSet.trainLabel))
        print("validating label(2) test",Counter(dataSet.validateLabel))

        self.model.fit(dataSet.trainX,dataSet.trainLabel)

        t1=time.time()

        #measure training result
        vpredict=self.predict(dataSet.validateX)
        #print(vpredict)
        vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        #print(vpredict)
        score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)
    def findPath(self):
        modelpath="../data/saved_ML_models/baseline/NBbayes-"+self.name+".pkl"
        return modelpath

class DecsionTree(ML_model):
    def initParameters(self):
        self.params={
            'criterion':"gini",
            'splitter':"best",
            'max_depth':5,
            'min_samples_split':2,
            'min_samples_leaf':1,
            'max_features':'auto',
        }
    def __init__(self):
        ML_model.__init__(self)
        self.initParameters()
    def predict(self,X):
        if  len(X)==0:
            return np.array([],dtype=np.int)
        if self.verbose>0:
            print(self.name," C4.5 is predicting")

        Y=self.model.predict_proba(X)
        return Y[:,1]
    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]
    def searchParameters(self,dataSet):
        print("searching for best parameters")

        selParas=[
            {'criterion':["gini",'entropy']},
            {'splitter':["best",'random']},
            {'max_depth':[i for i in range(3,10)]},
            {'min_samples_split':[i for i in range(2,10)]},
            {'min_samples_leaf':[i for i in range(1,10)]},
            {'max_features':[None,'sqrt','log2']},
        ]


        for i in range(len(selParas)):
            para=selParas[i]
            model=tree.DecisionTreeClassifier(**self.params)
            gsearch=GridSearchCV(model,para,scoring=metrics.make_scorer(metrics.accuracy_score))
            gsearch.fit(dataSet.trainX,dataSet.trainLabel)
            print("best para",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)

        self.model=tree.DecisionTreeClassifier(**self.params)

    def trainModel(self,dataSet):
        print("training")
        t0=time.time()

        self.searchParameters(dataSet)

        print("training label(2) test",Counter(dataSet.trainLabel))
        print("validating label(2) test",Counter(dataSet.validateLabel))

        self.model.fit(dataSet.trainX,dataSet.trainLabel)

        t1=time.time()

        #measure training result
        vpredict=self.predict(dataSet.validateX)
        #print(vpredict)
        vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        #print(vpredict)
        score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

    def findPath(self):
        modelpath="../data/saved_ML_models/baseline/DecisionTree"+self.name+".pkl"
        return modelpath


class KNN(ML_model):
    def __init__(self,K=5):
        ML_model.__init__(self)
        self.KN=K

    def predict(self,X):
        if  len(X)==0:
            return np.array([],dtype=np.int)
        if self.verbose>0:
            print(self.name," KNN-%d is predicting"%self.KN)

        Y=self.model.predict_proba(X)
        return Y[:,1]


    def trainModel(self,dataSet):
        print("training")
        t0=time.time()

        self.model=KNeighborsClassifier(self.KN)

        print("training label(2) test",Counter(dataSet.trainLabel))
        print("validating label(2) test",Counter(dataSet.validateLabel))

        self.model.fit(dataSet.trainX,dataSet.trainLabel)

        t1=time.time()

        #measure training result
        vpredict=self.predict(dataSet.validateX)
        #print(vpredict)
        vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        #print(vpredict)
        score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)
    def findPath(self):
        modelpath="../data/saved_ML_models/baseline/KNN-%d-"%self.KN+self.name+".pkl"
        return modelpath

def train(mode,method=0):

    tasktypes=SelectedTaskTypes.loadTaskTypes()["clustered"]
    #tasktypes=("global",)
    for tasktype in tasktypes:

        if method==0:
            model=KNN()
        elif method==1:
            model=DecsionTree()
        else:
            model=NBBayes()

        model.name=tasktype+"-classifier"+ModeTag[mode]

        data=loadData(tasktype,mode)
        #train model
        model.trainModel(data);model.saveModel()
        #measuer model
        model.loadModel()
        Y_predict2=model.predict(data.testX)
        showMetrics(Y_predict2,data,model.threshold)

def buildCrowdRex(tasktype,method=0):
    if method==0:
            model=KNN()
    elif method==1:
        model=DecsionTree()
    else:
        model=NBBayes()

    model.name=tasktype+"-classifier"+ModeTag[2]
    model.loadModel()
    return model

if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    from Utility import SelectedTaskTypes

    train(2)

