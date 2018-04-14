from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import ensemble
from sklearn import metrics
import time,json
from sklearn.model_selection import GridSearchCV

class EnsembleClassifier(ML_model):
    def initParameters(self):
        self.params={
            'n_estimators':10,
            'criterion':"gini",
            'max_depth':12,
            'min_samples_split':20,
            'min_samples_leaf':5,
            'max_features':"auto",
            'max_leaf_nodes':None,
            'bootstrap':False,
            'n_jobs':-1,
            'verbose':1,
            'class_weight':{0:1,1:5}
        }
    def __init__(self):
        ML_model.__init__(self)
        self.initParameters()
    def predict(self,X):
        print(self.name,"is predicting")
        Y=self.model.predict_proba(X)
        return Y[:,1]
    def loadConf(self):
        with open("../data/saved_ML_models/classifiers/config/"+self.name+".json","r") as f:
            paras=json.load(f)
            self.params=paras
    def saveConf(self):
        with open("../data/saved_ML_models/classifiers/config/"+self.name+".json","w") as f:
            json.dump(self.params,f)
    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]
    def searchParameters(self,dataSet):
        print("searching for best parameters")
        selParas=[
            {'n_estimators':[i for i in range(10,200,10)]},
            {'criterion':["gini","entropy"]},
            {'max_depth':[i for i in range(3,13)]},
            {'min_samples_split':[i for i in range(20,100,5)]},
            {'min_samples_leaf':[i for i in range(5,30,2)]},
            {'max_features':["auto","sqrt","log2",None]},
            {'class_weight':[{0:1,1:i} for i in range(1,7)]}
        ]
        for i in range(len(selParas)):
            para=selParas[i]
            model=ensemble.ExtraTreesClassifier(**self.params)
            gsearch=GridSearchCV(model,para)
            gsearch.fit(dataSet.trainX,dataSet.trainLabel)
            print("best para",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)
        self.saveConf()
        self.model=ensemble.ExtraTreesClassifier(**self.params)

    def trainModel(self,dataSet):
        print("training")
        t0=time.time()
        try:
            self.loadConf()
            self.model=ensemble.ExtraTreesClassifier(**self.params)
        except:
            print("loading configuration failed")
            self.searchParameters(dataSet)

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


if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    mode=0
    tasktype="Architecture"
    dnnmodel=EnsembleClassifier()
    dnnmodel.name=tasktype+"-classifier"+ModeTag[mode]

    data=loadData(tasktype,mode)
    #train model
    dnnmodel.trainModel(data);dnnmodel.saveModel()
    #measuer model
    dnnmodel.loadModel()
    Y_predict2=dnnmodel.predict(data.testX)
    showMetrics(Y_predict2,data,dnnmodel.threshold)

    #winners
    if mode==2:
        topKmetrics(Y_predict2,data)
