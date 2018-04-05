import pickle
import numpy as np
from Utility.personalizedSort import MySort
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

def topKAccuracy(Y_predict2,data,k):
    '''
    :return Y[i]=true if ith sample can intersect with each other in Y_predict[i] and Y_true[i]
                      else return false
    :param Y_predict2: a list of recommended entries
    :param data: the data set containing actual labels
    :return: boolean
    '''
    # measure top k accuracy

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

        ys=[]
        for i in range(len(predictY)):
            ys.append([i,predictY[i]])
        m_s=MySort(ys)
        m_s.compare_vec_index=-1
        ys=m_s.mergeSort()
        ys.reverse()
        ys=np.array(ys)
        #print(ys)
        predictY=ys[:,0]

        Y_predict.append(predictY)
        Y_true.append(np.where(trueY==0)[0])
        #print("predict",Y_predict)
        #print("true",Y_true)
        left=right
    #print(len(Y_true))
    #for i in range(len(Y_true)):
    #   print(Y_predict[i].shape,Y_true[i].shape)
    #test intersection
    Y=[]

    for i in range(len(Y_true)):
        if len(Y_true[i]==0):
            continue
        tag=False

        for ele in Y_predict[i][:k]:
            if ele in Y_true[i]:
                tag=True
                break

        Y.insert(i,tag)

    return np.array(Y)
