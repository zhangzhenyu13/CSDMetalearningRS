from ML_Models.Model_def import *
from keras import models,layers,optimizers,losses
from keras.utils import np_utils
import numpy as np
from sklearn import metrics
import time



class DNNCLassifier(ML_model):
    def __init__(self):
        ML_model.__init__(self)
        self.defineModelLayer()
    #define dnn layer
    def defineModelLayer(self):
        inputDim=415
        ouputDim=2
        model = models.Sequential()
        model.add(layers.Embedding(20000, 200))
        model.add(layers.LSTM(200, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(ouputDim, activation='sigmoid'))

        model.add(layers.Activation("softmax"))
        self.model=model

        opt = optimizers.Adagrad(lr=0.001)
        self.model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])

    def trainModel(self,dataSet):
        dataSet.trainLabel=np_utils.to_categorical(dataSet.trainLabel,num_classes=2)

        print(self.name+" training")
        t0=time.time()
        self.model.fit(x=dataSet.trainX,y=dataSet.trainLabel,epochs=5,batch_size=500)
        t1=time.time()
        loss,accuracy=self.model.evaluate(x=dataSet.trainX,y=dataSet.trainLabel,batch_size=10000)
        print("finished in %ds"%(t1-t0),"accuracy=%f"%accuracy,"loss=%f"%loss)

    def predict(self,X):
        print("dnn model predicting ")
        Y=self.model.predict(X,batch_size=10000)
        Y=np.argmax(Y,axis=1)
        print("finished predicting ",len(Y))
        return Y

    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/" + self.name + ".h5")

