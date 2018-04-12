import xgboost
from ML_Models.Model_def import ML_model
import time
import numpy as np
from sklearn import metrics

class XGBoostClassifier(ML_model):
    def __init__(self):
        ML_model.__init__(self)
        self.trainEpchos=150
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

        param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
        watchlist = [(dvalidate, 'eval'), (dtrain, 'train')]

        self.model=xgboost.train(params=param,dtrain=dtrain,
                                 num_boost_round=self.trainEpchos,evals=watchlist)


        ptrain = self.model.predict(dtrain, output_margin=True)
        ptest = self.model.predict(dvalidate, output_margin=True)
        dtrain.set_base_margin(ptrain)
        dvalidate.set_base_margin(ptest)

        print('this is result of running from initial prediction')
        self.model = xgboost.train(param, dtrain, 1, watchlist)

        t1=time.time()

        #measure training result
        vpredict=self.model.predict(xgboost.DMatrix(dataSet.validateX))
        vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        print(vpredict)
        score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)
