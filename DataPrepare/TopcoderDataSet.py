import _pickle as pickle
import numpy as np
import time
from Utility.TagsDef import *
from collections import Counter
from imblearn import under_sampling,over_sampling

class DataSetTopcoder:
    def __init__(self,testratio=0.2,validateratio=0.1):
        self.dataSet=None
        self.trainX=None
        self.trainLabel=None
        self.testX=None
        self.testLabel=None
        self.validateX=None
        self.validateLabel=None
        self.testRatio=testratio
        self.validateRatio=validateratio

    def setParameter(self,tasktype,mode,testMode=False):
        #file data
        self.tasktype=tasktype
        self.filepath="../data/TopcoderDataSet/"+ModeTag[mode].lower()+"HistoryBasedData/"+self.tasktype+"-user_task.data"
        if testMode:
            self.filepath="../data/TopcoderDataSet/"+ModeTag[mode].lower()+"HistoryBasedData/"\
                          +self.tasktype+"-test-user_task.data"
            print("\n Test Mode \n")
        self.testMode=testMode

    def indexDataPoint(self,taskids):
        id=taskids[0]
        IDIndex=[(id,0)]
        for i in range(1,len(taskids)):
            if taskids[i]!=id:
                id=taskids[i]
                IDIndex.append((id,i))
        return IDIndex

    def fetchData(self,files,key):
        '''
        :param files: the files that contain the data
        :param key: the data key
        :return: X, array like, containing the data
        '''
        print(self.tasktype+" fetching data,key="+key)
        VecX=[None for i in range(len(files))]
        self.fetchOne(files[0],key,0,VecX)
        X=VecX[0]
        for i in range(1,len(files)):
            file=files[i]
            self.fetchOne(file,key,i,VecX)
            X=np.concatenate((X,VecX[i]),axis=0)

        return X

    def fetchOne(self,file,key,index,vecX):
        '''
        fetch data segment from one file and put it into vecX
        :param file:
        :param key:
        :return:
        '''
        t0=time.time()
        data=None
        try:
            with open(file,"rb") as f:
                data=pickle.load(f)
        except:
            print(self.tasktype,"dataSegment Error",index)

        X=np.array(data[key])
        vecX[index]=X
        #print(X.shape)
        print(" fetched segment:",index,"in %ds"%(time.time()-t0))

    def loadData(self):

        print(self.tasktype,"loading data")
        with open(self.filepath,"rb") as f:
            self.dataSet=pickle.load(f)
        #print(self.dataSet)

        users=self.fetchData(self.dataSet,"users")
        tasks=self.fetchData(self.dataSet,"tasks")
        self.taskids=self.fetchData(self.dataSet,"taskids")
        #self.usernames=self.fetchData(self.dataSet,"usernames")
        X=np.concatenate((tasks,users),axis=1)

        self.IDIndex=self.indexDataPoint(taskids=self.taskids)
        tp=int(self.testRatio*len(self.IDIndex))
        if tp==len(self.IDIndex):
            self.testPoint=len(tasks)
        else:
            self.testPoint=self.IDIndex[tp][1]
        vp=int((self.testRatio+self.validateRatio)*len(self.IDIndex))
        if vp==len(self.IDIndex):
            self.validatePoint=len(tasks)
        else:
            self.validatePoint=self.IDIndex[vp][1]

        self.trainX=X[self.validatePoint:]
        self.validateX=X[self.testPoint:self.validatePoint]
        self.testX=X[:self.testPoint]

        print("total feature length for user(%d) and task(%d) is %d"%(len(users[0]),len(tasks[0]),len(X[0])))
        print("total tasks size=%d,test size=%d, validate size=%d"%(len(self.IDIndex),tp,vp-tp))
        print("loaded all the instances, size=%d"%len(self.taskids),
              "test point=%d, validate point=%d"%(self.testPoint,self.validatePoint))

    def ReSampling(self,data,labels,over_s=True):

        label_status=Counter(labels)
        print(self.tasktype,"data "+self.tasktype,label_status)

        featurelen=len(data[0])
        if 1 not in label_status.keys():
            x,y=np.zeros(shape=featurelen,dtype=np.int),1
        elif 0 not in label_status.keys():
            x,y=np.zeros(shape=featurelen,dtype=np.int),0
        else:
            x,y=None,None
        if x is not None:
            data=np.insert(data,0,x,0)
            labels=np.insert(labels,0,y,0)

        if len(label_status)<2:
            print(self.tasktype,"no need to resample")
            return data,labels
        if label_status[1]/label_status[0]<5. and label_status[1]/label_status[0]>0.2:
            print("data are not biased too much")
            return data,labels

        maxSamples=label_status[0]
        if label_status[1]>label_status[0]:
            maxSamples=label_status[1]
            resampling=over_sampling.ADASYN(ratio={1:maxSamples,0:int(0.4*maxSamples)})
        else:
            resampling=over_sampling.ADASYN(ratio={0:maxSamples,1:int(0.4*maxSamples)})

        try:
            data,labels=resampling.fit_sample(data,labels)
        except :
            print(self.tasktype,"resampling using random method")
            if over_s:
                resampling=over_sampling.RandomOverSampler()
            else:
                resampling=under_sampling.RandomUnderSampler()

            data,labels=resampling.fit_sample(data,labels)

        label_status=Counter(labels)
        print(self.tasktype,"sampling status=",label_status)

        return data,labels

class TopcoderReg(DataSetTopcoder):
    def __init__(self,tasktype,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
        self.setParameter(tasktype,0)

    def RegisterRegressionData(self):
        Y=self.fetchData(self.dataSet,"regists")
        self.registLabelRegression=Y
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

    def RegisterClassificationData(self):
        self.RegisterRegressionData()
        self.registLabelClassification=np.array(self.registLabelRegression>0,dtype=np.int)
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel=np.array(self.testLabel>0,dtype=np.int)

class TopcoderSub(DataSetTopcoder):
    def __init__(self,tasktype,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
        self.setParameter(tasktype,1)


    def constructTrainInstances(self):
        self.registLabelClassification=self.fetchData(self.dataSet,"regists")
        trainReg=self.registLabelClassification[self.validatePoint:]
        validateReg=self.registLabelClassification[self.testPoint:self.validatePoint]
        #print(len(trainReg),len(validateReg),len(self.trainLabel),len(self.validateLabel),
        #      len(self.trainX),len(self.validateX))
        indices=np.where(trainReg>0)[0]
        self.trainX=self.trainX[indices]
        self.trainLabel=self.trainLabel[indices]
        indices=np.where(validateReg>0)[0]
        self.validateX=self.validateX[indices]
        self.validateLabel=self.validateLabel[indices]
        print("after refactoring, train size=%d,validate size=%d,test size=%d"%(
            len(self.trainLabel),len(self.validateLabel),len(self.testLabel)))

    def SubmitRegressionData(self):

        Y=self.fetchData(self.dataSet,"submits")
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]
        self.constructTrainInstances()

    def SubmitClassificationData(self):
        self.SubmitRegressionData()
        self.trainLabel=np.array(self.trainLabel>0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel>0,dtype=np.int)
        self.testLabel=np.array(self.testLabel>0,dtype=np.int)

class TopcoderWin(DataSetTopcoder):

    def __init__(self,tasktype,testratio=0.2,validateratio=0.1):
        DataSetTopcoder.__init__(self,testratio=testratio,validateratio=validateratio)
        self.setParameter(tasktype,2)

    def constructTrainInstances(self):
        self.registerLabelClassification=self.fetchData(self.dataSet,"regists")
        self.submitLabelClassification=self.fetchData(self.dataSet,"submits")
        trainSub=self.submitLabelClassification[self.validatePoint:]
        validateSub=self.submitLabelClassification[self.testPoint:self.validatePoint]
        indices=np.where(trainSub>0)[0]
        self.trainX=self.trainX[indices]
        self.trainLabel=self.trainLabel[indices]
        indices=np.where(validateSub>0)[0]
        if len(indices)==0:
            self.validateLabel=self.trainLabel[-2:-1]
            self.validateX=self.trainX[-2:-1]
        else:
            self.validateX=self.validateX[indices]
            self.validateLabel=self.validateLabel[indices]

        print("after refactoring, train size=%d,validate size=%d,test size=%d"%(
            len(self.trainLabel),len(self.validateLabel),len(self.testLabel)))

    def WinRankData(self):
        Y=self.fetchData(self.dataSet,"ranks")
        self.trainLabel=Y[self.validatePoint:]
        self.validateLabel=Y[self.testPoint:self.validatePoint]
        self.testLabel=Y[:self.testPoint]

        if self.testMode==False:
            self.constructTrainInstances()

    def WinClassificationData(self):
        self.WinRankData()
        self.trainLabel=np.array(self.trainLabel==0,dtype=np.int)
        self.validateLabel=np.array(self.validateLabel==0,dtype=np.int)
        self.testLabel=np.array(self.testLabel==0,dtype=np.int)

if __name__ == '__main__':
    from Utility import SelectedTaskTypes
    tasktypes=SelectedTaskTypes.loadTaskTypes()["keeped"]
    #tasktypes=("global",)
    for tasktype in tasktypes:
        data=TopcoderWin(tasktype)
        data.loadData()
        print(tasktype,data.validatePoint,data.testPoint)
