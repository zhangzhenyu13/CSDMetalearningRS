from ML_Models.Model_def import *
from keras import models,layers,optimizers
import numpy as np
from sklearn import metrics
import time
class DNNCLassifier(ML_model):
    def __init__(self,dataSet):
        ML_model.__init__(self)
        self.dataSet=dataSet
        self.defineModelLayer()
    #define dnn layer
    def defineModelLayer(self):
        inputDim=len(self.dataSet.trainX[0])
        ouputDim=2
        self.model=models.Sequential()
        self.model.add(layers.Dense(units=2048,input_dim=inputDim))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=1800))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=1596))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=1024))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=896))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=768))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=512))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=256))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=128))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.Dense(units=64))
        self.model.add(layers.Dense(units=2))
        self.model.add(layers.Activation("softmax"))

        rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=["accuracy"])

    def trainModel(self):
        print("dnn classifier training")
        t0=time.time()
        self.model.fit(x=self.dataSet.trainX,y=self.dataSet.trainLabel,epochs=1,batch_size=100)
        t1=time.time()
        loss,accuracy=self.model.evaluate(x=self.dataSet.trainX,y=self.dataSet.trainLabel,batch_size=200)
        print("finished in %ds"%(t1-t0),"accuracy=%f"%accuracy,"loss=%f"%loss)

    def predict(self,X):
        print("dnn model predicting ")
        Y=self.model.predict(X,batch_size=200)
        print("finished predicting ",len(Y),np.shape(Y))
        return Y
    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/" + self.name + ".h5")
        self.model = None

if __name__ == '__main__':
    data=DataSetTopcoder()
    data.CommitClassificationData()
    data.trainLabel,d1=labeltoVector(data.trainLabel,(0,1))
    data.testLabel,d2=labeltoVector(data.testLabel,(0,1))
    model=DNNCLassifier(dataSet=data)
    model.name = "DNN _lassifier"
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict2 = model.predict(data.testX)
    Y_predict2=VectortoLabel(Y_predict2,d2)
    print("test score=%f" % (metrics.accuracy_score(data.testLabel, Y_predict2, normalize=True)))
    print()
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel, Y_predict2))