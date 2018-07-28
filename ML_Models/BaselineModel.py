from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import tree,svm,naive_bayes,ensemble
from sklearn import metrics
import time,json
from sklearn.model_selection import GridSearchCV
import warnings
from ML_Models.UserMetrics import TopKMetrics

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

class RandForest(ML_model):
    def initParameters(self):
        self.params={
            "n_estimators":10,
            "criterion":"gini",
            "max_depth":None,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "min_weight_fraction_leaf":0.,
            "max_features":"auto",
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.,
            "min_impurity_split":None,
            "n_jobs":-1
        }
    def __init__(self):
        ML_model.__init__(self)
        self.initParameters()
    def predict(self,X):
        if  len(X)==0:
            return np.array([],dtype=np.int)
        if self.verbose>0:
            print(self.name," RF is predicting")

        Y=self.model.predict_proba(X)
        return Y[:,1]
    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]
    def searchParameters(self,dataSet):
        print("searching for best parameters")

        selParas=[
            {"n_estimators":[i for i in range(10,101,10)]},
            {"criterion":["gini","entropy"]},
            {"max_depth":[i for i in range(5,12)]},
            {"min_samples_split":[2,5,7,10]},
            {"min_samples_leaf":[1,2,3,4,5]},
            {"max_features":["log2","sqrt"]},
            {"min_impurity_decrease":[i/100.0 for i in range(0,100,5)]}
        ]


        for i in range(len(selParas)):
            para=selParas[i]
            model=ensemble.RandomForestClassifier(**self.params)
            gsearch=GridSearchCV(model,para,scoring=metrics.make_scorer(metrics.accuracy_score))
            gsearch.fit(dataSet.trainX,dataSet.trainLabel)
            print("best para",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)

        self.model=ensemble.RandomForestClassifier(**self.params)

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
        modelpath="../data/saved_ML_models/baseline/RandomForest"+self.name+".pkl"
        return modelpath

if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    from Utility import SelectedTaskTypes
    tasktypes=SelectedTaskTypes.loadTaskTypes()
    for tasktype in tasktypes["keeped"]:
        for mode in (2,):

            if "Code" in tasktype or "Assembly" in tasktype or "First2Finish" in tasktype:
                continue
            dnnmodel=RandForest()
            dnnmodel.name=tasktype+"-classifier"+ModeTag[mode]

            #train model
            try:
                dnnmodel.loadModel()
            except:
                print("model for "+tasktype+" does not exit, train mow")
                data=loadData(tasktype,mode)
                dnnmodel.trainModel(data);dnnmodel.saveModel()
            #measuer model
            #dnnmodel.loadModel()
            #Y_predict2=dnnmodel.predict(data.testX)
            #showMetrics(Y_predict2,data,dnnmodel.threshold)


