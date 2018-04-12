import xgboost
from ML_Models.Model_def import ML_model
import time
import numpy as np
from sklearn import metrics

class XGBoostClassifier(ML_model):
    def getParameters(self):
        params={
            'booster':'gbtree',
            'objective':'binary:logistic', #多分类的问题
            'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth':12, # 构建树的深度，越大越容易过拟合
            'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample':0.7, # 随机采样训练样本
            'colsample_bytree':0.7, # 生成树时进行的列采样
            'min_child_weight':3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.007, # 如同学习率
            'seed':1000,
            'nthread':7,# cpu 线程数
            'eval_metric': 'auc'
            }
        return params
    def __init__(self):
        ML_model.__init__(self)
        self.trainEpchos=5000
        self.threshold=0.5
    def predict(self,X):
        print(self.name,"XGBoost model is predicting")
        inputTD=xgboost.DMatrix(data=X)
        Y=self.model.predict(inputTD)
        Y=np.array(Y>self.threshold,dtype=np.int)
        return Y
    def trainModel(self,dataSet):
        print("training")
        t0=time.time()

        dtrain=xgboost.DMatrix(data=dataSet.trainX,label=dataSet.trainLabel)
        dvalidate=xgboost.DMatrix(data=dataSet.validateX,label=dataSet.validateLabel)

        param =self.getParameters()#= {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
        watchlist = [(dvalidate, 'eval'), (dtrain, 'train')]

        self.model=xgboost.train(params=param,dtrain=dtrain,
                                 num_boost_round=self.trainEpchos,evals=watchlist,
                                 early_stopping_rounds=20)


        ptrain = self.model.predict(dtrain, output_margin=True)
        ptest = self.model.predict(dvalidate, output_margin=True)
        dtrain.set_base_margin(ptrain)
        dvalidate.set_base_margin(ptest)

        print('this is result of running from initial prediction')
        self.model = xgboost.train(param, dtrain, 1, watchlist)

        t1=time.time()

        #measure training result
        vpredict=self.model.predict(xgboost.DMatrix(dataSet.validateX),ntree_limit=self.model.best_ntree_limit)
        print(vpredict)
        vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        print(vpredict)
        score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

    def findPath(self):
        modelpath="../data/saved_ML_models/boosts/"+self.name+".pkl"
        return modelpath
