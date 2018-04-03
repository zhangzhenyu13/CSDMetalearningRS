import pickle
import numpy as np

class ML_model:
    def __init__(self):
        self.model=None
        self.name=""
        self.dataSet=None

    def predict(self,X):
        '''
        predict the result based on given X
        :param X: input samples,(n,D)
        :return: given result, class or a real num
        '''
    def trainModel(self):
        pass

    def findPath(self):
        modelpath="../data/saved_ML_models/classifiers/"+self.name+".pkl"
        return modelpath
    def loadModel(self):
        with open(self.findPath(),"rb") as f:
            data=pickle.load(f)
            self.model=data["model"]
            self.name=data["name"]
    def saveModel(self):
        with open(self.findPath(),"wb") as f:
            data={}
            data["model"]=self.model
            data["name"]=self.name
            pickle.dump(data,f)

def topKAccuracy(Y_predict2,data,k=5):
    '''
    :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                      else return false
    :param Y_predict2: a list of recommended entries
    :param data: the data set containing actual labels
    :return: boolean
    '''
    # measure top k accuracy
    print("predicting top k accuracy of",data.tasktype)
    # batch data into task centered array
    ids=data.taskids[:data.testPoint]
    indexData=data.indexDataPoint(ids)
    left=indexData[0][1]

    Y_predict=[]
    Y_true=[]

    for step in range(1,len(indexData)):
        right=indexData[step][1]

        trueY=data.testLabel[left:right]
        predictY=Y_predict2[left:right]
        Y_predict.append(predictY)
        Y_true.append(trueY[np.where(trueY==0)])
    Y_predict=np.array(Y_predict)
    Y_true=np.array(Y_true)
    print(Y_predict.shape,Y_true.shape)
    #test intersection
    Y=np.zeros(shape=(len(Y_true)),dtype=np.bool)

    for i in range(len(Y)):
        tag=False

        for ele in Y_predict[i][:k]:
            if ele in Y_true[i]:
                tag=True
                break

        Y[i]=tag

    return Y
