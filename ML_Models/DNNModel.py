from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from keras import models,layers,optimizers,losses
from keras.utils import np_utils
import numpy as np
from sklearn import metrics
import time
import matplotlib.pyplot as plt

class DNNRegression(ML_model):
    def __init__(self,dataSet):
        ML_model.__init__(self)
        self.dataSet=dataSet
        self.defineModellayer()
    def defineModellayer(self):
        inputDim = len(self.dataSet.trainX[0])
        ouputDim = 1
        self.model = models.Sequential()
        self.model.add(layers.Dense(units=2048, input_dim=inputDim))
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

        self.model.add(layers.Dense(units=ouputDim))

        opt = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=opt, loss=losses.mean_squared_error)
    def trainModel(self):
        print(self.name + " training")
        t0 = time.time()
        self.model.fit(x=self.dataSet.trainX, y=self.dataSet.trainLabel, epochs=1, batch_size=500)
        t1 = time.time()

        mse = self.model.evaluate(x=self.dataSet.trainX, y=self.dataSet.trainLabel, batch_size=10000)
        print("finished in %ds" % (t1 - t0), "mse=", mse)
    def predict(self,X):
        print("dnn model predicting ")
        Y = self.model.predict(X, batch_size=10000)
        print("finished predicting ", len(Y),Y.shape)
        return Y
    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/" + self.name + ".h5")
        self.model = None

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
        self.model.add(layers.Dense(units=ouputDim))
        self.model.add(layers.Activation("softmax"))

        opt = optimizers.Adagrad(lr=0.001)
        self.model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])

    def trainModel(self):
        print(self.name+" training")
        t0=time.time()
        self.model.fit(x=self.dataSet.trainX,y=self.dataSet.trainLabel,epochs=5,batch_size=500)
        t1=time.time()
        loss,accuracy=self.model.evaluate(x=self.dataSet.trainX,y=self.dataSet.trainLabel,batch_size=10000)
        print("finished in %ds"%(t1-t0),"accuracy=%f"%accuracy,"loss=%f"%loss)

    def predict(self,X):
        print("dnn model predicting ")
        Y=self.model.predict(X,batch_size=200)
        print("finished predicting ",len(Y))
        return Y
    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/" + self.name + ".h5")
        self.model = None

def testClassification(data):
    data.CommitClassificationData()
    data.trainLabel=np_utils.to_categorical(data.trainLabel,num_classes=2)

    model = DNNCLassifier(dataSet=data)
    model.name = data.tasktype+"-DNN _classifier"
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict2 = model.predict(data.testX)
    Y_predict2=np.argmax(Y_predict2,axis=1)

    print("test score=%f" % (metrics.accuracy_score(data.testLabel, Y_predict2, normalize=True)))
    print()

    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel, Y_predict2))

def testRegression(data):
    data.CommitRegressionData()
    model=DNNRegression(dataSet=data)
    model.name="DNN_Regression"
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict1=model.predict(data.testX)
    print("test score=%f" % (metrics.mean_squared_error(data.testLabel, Y_predict1)))
    print()
    showData(Y_predict1,data.testLabel,"sub num")
def showData(Y_predict,Y_true,content):
    X=np.arange(0,len(Y_predict))
    plt.figure("test and real")
    plt.plot(X,Y_predict,color="r")
    plt.plot(X,Y_true,color="g")
    plt.xlabel("intsances")
    plt.ylabel(content)
    plt.title("test and real "+content)
    plt.show()
if __name__ == '__main__':
    data=TopcoderSub(testratio=0.2,validateratio=0)
    data.setParameter(tasktype="Architecture",choice=1)
    data.loadData()
    testClassification(data)
    #testRegression(data)
