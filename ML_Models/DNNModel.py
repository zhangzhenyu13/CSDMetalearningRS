from ML_Models.Model_def import *
from keras import models,layers,optimizers,losses
import numpy as np
import time,json
from Utility.TagsDef import ModeTag
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from ML_Models.UserMetrics import TopKMetrics
import warnings

warnings.filterwarnings("ignore")

#create model
def createDNN(dp=0.5):
    inputDim=126#user:60, task:66
    ouputDim=1
    DNNmodel=models.Sequential()
    DNNmodel.add(layers.Dense(units=96,input_shape=(inputDim,),activation="relu"))
    DNNmodel.add(layers.Dense(units=96,activation="relu"))
    DNNmodel.add(layers.Dense(units=64,activation="relu"))
    DNNmodel.add(layers.Dense(units=64,activation="relu"))
    DNNmodel.add(layers.Dense(units=32,activation="relu"))
    DNNmodel.add(layers.Dense(units=32,activation="relu"))
    DNNmodel.add(layers.Dense(units=16,activation="relu"))
    DNNmodel.add(layers.Dropout(dp))
    DNNmodel.add(layers.Dense(units=ouputDim,activation="sigmoid"))

    opt = optimizers.Adadelta()
    DNNmodel.compile(optimizer=opt,loss=losses.mean_squared_error)
    return DNNmodel

#model
class DNNCLassifier(ML_model):
    def initParameters(self):
        self.params={

            'dp':0.5,
            'verbose':0,
        }
    def __init__(self):
        ML_model.__init__(self)
        self.initParameters()

    def loadConf(self):
        with open("../data/saved_ML_models/dnns/config/"+self.name+".json","r") as f:
            paras=json.load(f)
        for k in paras.keys():
            self.params[k]=paras[k]

    def saveConf(self):
        with open("../data/saved_ML_models/dnns/config/"+self.name+".json","w") as f:
            json.dump(self.params,f)
    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]

    def searchParameters(self,dataSet):
        print("searching for best parameters")

        selParas=[
            {'dp':[i/10 for i in range(3,8)]}
        ]

        for i in range(len(selParas)):
            para=selParas[i]
            model=KerasRegressor(createDNN,**self.params)
            gsearch=GridSearchCV(model,para)
            gsearch.fit(dataSet.trainX,dataSet.trainLabel)
            print("best para",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)

        self.saveConf()
        paras=self.params
        self.model=createDNN(paras['dp'])

    def trainModel(self,dataSet):
        print(self.name+" training")
        t0=time.time()
        try:
            self.loadConf()
            paras=self.params
            self.model=createDNN(paras['dp'])
        except:
            print("loading configuration failed")
            self.searchParameters(dataSet)

        self.model.fit(dataSet.trainX,dataSet.trainLabel,verbose=0,epochs=5,batch_size=500)

        t1=time.time()
        mse=self.model.evaluate(dataSet.validateX,dataSet.validateLabel,verbose=0,batch_size=10000)
        print("finished in %ds"%(t1-t0),"mse=",mse)

    def predict(self,X):
        if self.verbose>0:
            print(self.name,"(DNN) is predicting ")
        Y=self.model.predict(X,batch_size=10000,verbose=0)
        #print("finished predicting ",len(Y))
        return Y

    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/dnns/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/dnns/" + self.name + ".h5")

if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    from Utility import SelectedTaskTypes
    tasktypes=SelectedTaskTypes.loadTaskTypes()["clustered"]
    tasktypes=("global",)
    for tasktype in tasktypes:
        for mode in (0,1):

            dnnmodel=DNNCLassifier()
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
                mymetric=TopKMetrics(data.tasktype)

                topKmetrics(mymetric,Y_predict2,data)
        print()
