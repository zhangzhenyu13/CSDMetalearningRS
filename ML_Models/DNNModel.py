from ML_Models.Model_def import *
from keras import models,layers,optimizers,losses
from keras.utils import np_utils
import numpy as np
import time
from Utility.TagsDef import ModeTag

class DNNCLassifier(ML_model):
    def __init__(self):
        ML_model.__init__(self)
        self.defineModelLayer()
    #define dnn layer
    def defineModelLayer(self):
        inputDim=215
        ouputDim=2
        x=layers.Input(shape=(inputDim,))

        #model1
        model1=layers.Dense(400,activation="relu")(x)
        model1=layers.Dense(160, activation="relu")(model1)
        model1=layers.Dropout(0.5)(model1)

        model3=layers.Dense(ouputDim,activation="softmax")(model1)

        #final model
        self.model=models.Model(inputs=[x],outputs=[model3])

        opt = optimizers.Adagrad(lr=0.001)
        self.model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])

    def trainModel(self,dataSet):
        dataSet.trainLabel=np_utils.to_categorical(dataSet.trainLabel,num_classes=2)
        dataSet.validateLabel=np_utils.to_categorical(dataSet.validateLabel,num_classes=2)
        X=np.concatenate((dataSet.trainX,dataSet.validateX),axis=0)
        Y=np.concatenate((dataSet.trainLabel,dataSet.validateLabel),axis=0)

        print(self.name+" training")
        t0=time.time()
        self.model.fit(x=X,y=Y,epochs=5,batch_size=256)
        t1=time.time()
        loss,accuracy=self.model.evaluate(x=X,y=Y,batch_size=10000)
        print("finished in %ds"%(t1-t0),"accuracy=%f"%accuracy,"loss=%f"%loss)

    def predict(self,X):
        print(self.name,"(DNN) is predicting ")
        Y=self.model.predict(X,batch_size=10000)
        Y=np.argmax(Y,axis=1)
        print("finished predicting ",len(Y))
        return Y

    def loadModel(self):
        self.model=models.load_model("../data/saved_ML_models/dnns/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("../data/saved_ML_models/dnns/" + self.name + ".h5")

if __name__ == '__main__':
    from ML_Models.XGBTuning import loadData,showMetrics
    mode=0
    dnnmodel=DNNCLassifier()
    dnnmodel.name="global-classifier"+ModeTag[mode]

    data=loadData("global",0)

    #measuer model
    Y_predict2=dnnmodel.predict(data.testX)
    showMetrics(Y_predict2,dnnmodel.threshold)
