from ML_Models.Model_def import *
from keras import models,layers,optimizers,losses
import numpy as np
import time,json
from Utility.TagsDef import ModeTag
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#create model
def createDNN(l1=64,l2=64,l3=64,l4=32,l5=32,l6=16,dp=0.5):
    inputDim=126#user:60, task:66
    ouputDim=1
    DNNmodel=models.Sequential()
    DNNmodel.add(layers.Dense(units=l1,input_shape=(inputDim,),activation="relu"))
    DNNmodel.add(layers.Dense(units=l2,activation="relu"))
    DNNmodel.add(layers.Dense(units=l3,activation="relu"))
    DNNmodel.add(layers.Dense(units=l4,activation="relu"))
    DNNmodel.add(layers.Dense(units=l5,activation="relu"))
    DNNmodel.add(layers.Dense(units=l6,activation="relu"))
    DNNmodel.add(layers.Dropout(dp))
    DNNmodel.add(layers.Dense(units=ouputDim,activation="sigmoid"))

    opt = optimizers.Adagrad()
    DNNmodel.compile(optimizer=opt,loss=losses.mean_squared_error,metrics=["accuracy"])
    return DNNmodel

#model
class DNNCLassifier(ML_model):
    def initParameters(self):
        self.params={
            'l1':64,
            'l2':64,
            'l3':64,
            'l4':32,
            'l5':32,
            'l6':16,
            'dp':0.5,
            'verbose':0
        }
    def __init__(self):
        ML_model.__init__(self)
        self.initParameters()

    def loadConf(self):
        with open("../data/saved_ML_models/dnns/config/"+self.name+".json","r") as f:
            paras=json.load(f)
            self.params=paras
    def saveConf(self):
        with open("../data/saved_ML_models/dnns/config/"+self.name+".json","w") as f:
            json.dump(self.params,f)
    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]

    def searchParameters(self,dataSet):
        print("searching for best parameters")

        selParas=[
            {'l1':[i for i in range(64,128,8)]},
            {'l2':[i for i in range(64,128,8)]},
            {'l3':[i for i in range(64,96,8)]},
            {'l4':[i for i in range(32,64,8)]},
            {'l5':[i for i in range(32,64,8)]},
            {'l6':[i for i in range(16,48,8)]},
            {'dp':[i/10 for i in range(4,10)]}
        ]


        for i in range(len(selParas)):
            para=selParas[i]
            model=KerasRegressor(createDNN,**self.params)
            gsearch=GridSearchCV(model,para)
            gsearch.fit(dataSet.trainX,dataSet.trainLabel)
            print("best para",gsearch.best_params_)
            self.updateParameters(gsearch.best_params_)

        self.saveConf()
        self.model=KerasRegressor(createDNN,**self.params)

    def trainModel(self,dataSet):
        print(self.name+" training")
        t0=time.time()
        try:
            self.loadConf()
            self.model=KerasRegressor(createDNN,**self.params)
        except:
            print("loading configuration failed")
            self.searchParameters(dataSet)

        self.model.fit(dataSet.trainX,dataSet.trainLabel,)


        t1=time.time()
        loss=self.model.score(dataSet.validateX,dataSet.validateLabel)
        Y_predict=self.model.predict(dataSet.validateX)
        Y_predict=np.array(Y_predict>self.threshold,dtype=np.int)
        accuracy=metrics.accuracy_score(dataSet.validateLabel,Y_predict)
        print("finished in %ds"%(t1-t0),"accuracy=%f"%accuracy,"loss=%f"%loss)

    def predict(self,X):
        print(self.name,"(DNN) is predicting ")
        Y=self.model.predict(X,batch_size=10000,verbose=0)
        print("finished predicting ",len(Y))
        return Y

    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/dnns/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/dnns/" + self.name + ".h5")

if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    from Utility import SelectedTaskTypes
    tasktypes=SelectedTaskTypes.loadTaskTypes()["keeped"]
    #tasktypes=("global",)
    for tasktype in tasktypes:
        for mode in (0,1,2):

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
                topKmetrics(Y_predict2,data)
        print()
