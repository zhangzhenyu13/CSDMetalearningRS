import xgboost
from ML_Models.Model_def import ML_model
import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


class XGBoostClassifier(ML_model):

    def initParameters(self):
        self.params={
            'booster':'gbtree',
            'objective':'binary:logistic', #多分类的问题
            'n_estimators':350,
            'learning_rate':0.25,
            'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth':5, # 构建树的深度，越大越容易过拟合
            'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample':0.7, # 随机采样训练样本
            'colsample_bytree':0.7, # 生成树时进行的列采样
            'min_child_weight':1,
            #这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言,
            #假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.007, # 如同学习率
            'seed':1000,
            'reg_alpha':100,
            'nthread':8,# cpu 线程数
            'eval_metric': 'error@'+str(self.threshold)
            }

    def updateParameters(self,new_paras):
        for k in new_paras:
            self.params[k]=new_paras[k]

        self.threshold=eval(self.params['eval_metric'][6:])

    def loadConf(self):
        with open("../data/saved_ML_models/boosts/config/"+self.name+".json","r") as f:
            import json
            paras=json.load(f)
            self.params=paras

        self.threshold=eval(self.params['eval_metric'][6:])

    def __init__(self):
        ML_model.__init__(self)
        self.trainEpchos=5000
        self.threshold=0.5
        self.initParameters()

    def predict(self,X):
        print(self.name,"XGBoost model is predicting")
        InputData=xgboost.DMatrix(data=X)
        Y=self.model.predict(InputData,ntree_limit=self.model.best_ntree_limit)

        return Y

    def navieTrain(self,dataSet):
        print(" navie training")
        t0=time.time()

        dtrain=xgboost.DMatrix(data=dataSet.trainX,label=dataSet.trainLabel)
        dvalidate=xgboost.DMatrix(data=dataSet.validateX,label=dataSet.validateLabel)

        watchlist = [(dvalidate, 'eval'), (dtrain, 'train')]

        #begin to search best parameters
        self.model=xgboost.train(params=self.params,dtrain=dtrain,
                                 num_boost_round=self.trainEpchos,evals=watchlist,
                                 early_stopping_rounds=20)

        t1=time.time()

        #measure training result
        vpredict=self.model.predict(xgboost.DMatrix(dataSet.validateX),ntree_limit=self.model.best_ntree_limit)
        #print(vpredict)
        vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        #print(vpredict)
        score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

    def searchParameters(self,dataSet):
        print(" search training")
        t0=time.time()

        paraSelection=[
            {'n_estimators':[i for i in range(100,500,50)],'learning_rate':[i/100 for i in range(5,30,5)]},
            {'max_depth':[i for i in range(1,7)],'min_child_weight':[i for i in range(1,6)]},
            {'gamma':[i/10.0 for i in range(0,5)]},
            {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]},
            {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]},
            {'eval_metric':['error@'+str(i/10) for i in range(3,8)]}
        ]

        for i in range(len(paraSelection)):

            para1=paraSelection[i]
            print("step#%d"%(i+1),"select paras:",para1.keys())

            self.model=xgboost.XGBClassifier(**self.params)
            gsearch=GridSearchCV(self.model,para1)
            gsearch.fit(dataSet.trainX,dataSet.trainLabel)
            print("best paras",gsearch.best_params_)
            print("best score",gsearch.best_score_)
            self.updateParameters(gsearch.best_params_)

        print("save params of", dataSet.tasktype,"para search finished in %ds"%(time.time()-t0))
        with open("../data/saved_ML_models/boosts/config/"+self.name+".json","w") as f:
            import json
            json.dump(self.params,f)

    def trainModel(self,dataSet):

        reSearch=True
        loadedConf=True

        #procedure 1=>search best parameters
        try:
            self.loadConf()
        except:
            print("configuration of "+self.name+" loading failed")
            loadedConf=False

        if reSearch or loadedConf==False:
            self.searchParameters(dataSet)

        #procedure 2=> train true model
        self.navieTrain(dataSet)

    def findPath(self):
        modelpath="../data/saved_ML_models/boosts/"+self.name+".pkl"
        return modelpath

if __name__ == '__main__':
    from Utility.TagsDef import ModeTag
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    mode=0
    tasktype="Architecture"
    model=XGBoostClassifier()
    model.name=tasktype+"-classifier"+ModeTag[mode]

    data=loadData(tasktype,mode)
    model.trainModel(data);model.saveModel()

    model.loadModel()
    Y_predict2=model.predict(data.testX)
    showMetrics(Y_predict2,data,model.threshold)

    if mode==2:
        topKmetrics(Y_predict2,data)
