from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import metrics
from sklearn import ensemble
import time
from sklearn.model_selection import GridSearchCV
import warnings
#from ML_Models.Model_def import Maskdata
#Maskdata.initMasks(130)
#mask=Maskdata.maskLanguage
warnings.filterwarnings("ignore")

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

class DCWDS:
    def __init__(self,regLearner,subLearner,winLearner):
        self.regLearner=regLearner
        self.subLearner=subLearner
        self.winLearner=winLearner
    def predict(self,X):
        regs=self.regLearner.predict(X)
        subs=self.subLearner.predict(X)
        wins=self.winLearner.predict(X)
        results=np.ones_like(regs)
        results[regs<0.5]=0
        results[subs<0.5]=0
        results[wins<0.5]=0
        #a developer can win only if he satisfy the judgement condictions in the original paper workflow
        # thus such developer is given winning change tagged as 1
        return results

def buildDCW_DS(tasktype):
    regL=RandForest()
    regL.name=tasktype+"-classifier"+ModeTag[0]

    subL=RandForest()
    subL.name=tasktype+"-classifier"+ModeTag[1]

    winL=RandForest()
    winL.name=tasktype+"-classifier"+ModeTag[2]

    return DCWDS(regL,subL,winL)

def train(mode):

    tasktypes=SelectedTaskTypes.loadTaskTypes()
    #tasktypes=("global",)
    for tasktype in tasktypes:


        model=RandForest()
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

    #train registration
    train(0)
    #train submission
    train(1)
    #train winner?
    train(2)
