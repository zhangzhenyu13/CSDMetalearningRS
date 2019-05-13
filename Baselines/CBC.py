from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import tree,naive_bayes
from sklearn import metrics
import time
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.pipeline import Pipeline
from ML_Models.ClusteringModel import ClusteringModel
warnings.filterwarnings("ignore")


class SVMClassifier(ML_model):

    def __init__(self):
        ML_model.__init__(self)
    def predict(self,X):
        if  len(X)==0:
            return np.array([],dtype=np.int)
        if self.verbose>0:
            print(self.name," SVM is predicting")

        Y=self.model.predict_proba(X)
        return Y[:,1]



    def trainModel(self,dataSet):
        print("training")
        t0=time.time()

        self.model=svm.SVC(probability=True)


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
        modelpath="../data/saved_ML_models/baseline/SVM-"+self.name+".pkl"
        return modelpath



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


class CBCModel:
    def __init__(self,learner,clustering):
        self.model=learner
        self.clustering=clustering
    def predict(self,X,clustered=False):
        if clustered:
            Y=self.model.predict_proba(X)
            return Y[:,1]

        predictor=Pipeline([('clustering',self.clustering),('classifier',self.model)])
        Y=predictor.predict_proba(X)
        return Y[:,1]

def buildCBC(tasktype,classifierNo=0):
    if classifierNo==0:
        classifier=DecsionTree()
    else:
        classifier=NBBayes()

    classifier.name=tasktype+"-classifier"+ModeTag[2]
    classifier.loadModel()

    clustering=ClusteringModel()

    return CBCModel(classifier,clustering)

def train(mode):
    #CBC employs an clustering+ classification style method to predict winners
    # the clustering is switft down to the task selection and data preprocessing container class
    tasktypes=SelectedTaskTypes.loadTaskTypes()["clustered"]
    #tasktypes=("global",)
    for tasktype in tasktypes:


        model=DecsionTree()
        model.name=tasktype+"-classifier"+ModeTag[mode]

        data=loadData(tasktype,mode)
        #train model
        model.trainModel(data);model.saveModel()
        #measuer model
        model.loadModel()
        Y_predict2=model.predict(data.testX)
        showMetrics(Y_predict2,data,model.threshold)


if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    from Utility import SelectedTaskTypes

    train(2)
