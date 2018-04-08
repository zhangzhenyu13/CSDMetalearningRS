from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import svm,linear_model,naive_bayes,tree
from sklearn import ensemble
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import multiprocessing
#model container
class TraditionalRegressor(ML_model):
    def __init__(self,regressor):
        ML_model.__init__(self)
        self.model=regressor
    def predict(self,X):
        print(self.name,"is predicting")
        Y=self.model.predict(X)
        return Y
    def trainModel(self):
        print("training")
        t0=time.time()
        self.model.fit(self.dataSet.trainX,self.dataSet.trainLabel)
        t1=time.time()
        mse=metrics.mean_squared_error(self.dataSet.trainLabel,self.model.predict(self.dataSet.trainX))
        print("model",self.name,"training finished in %ds"%(t1-t0),"train mse=%f"%mse)

class TraditionalClassifier(ML_model):
    def ModelTuning(self):
        candite_selection = {
            "RandomFrorest": ensemble.RandomForestClassifier(),
            "ExtraForest": ensemble.ExtraTreesClassifier(),
            #"AdaBoost": ensemble.AdaBoostClassifier(),
            #"GradientBoost": ensemble.GradientBoostingClassifier(),
            #"SVM": svm.SVC(C=0.9)
        }
        return candite_selection
    def __init__(self):
        ML_model.__init__(self)
    def predict(self,X):
        print(self.name,"is predicting")
        Y=self.model.predict(X)
        return Y
    def trainModel(self):
        print("training")
        t0=time.time()
        candidate_model=self.ModelTuning()
        max_acc=0
        sel_model=None
        for key in candidate_model.keys():
            self.model=candidate_model[key]
            if self.model is not None:
                self.model.fit(self.dataSet.trainX,self.dataSet.trainLabel)
                v_predict=self.model.predict(self.dataSet.validateX)

                acc=metrics.accuracy_score(self.dataSet.validateLabel,v_predict)
                print(key,acc)
                if acc>max_acc:
                    sel_model=self.model
                    max_acc=acc
        self.model=sel_model

        trainData=np.concatenate((self.dataSet.trainX,self.dataSet.validateX),axis=0)
        trainLabel=np.concatenate((self.dataSet.trainLabel,self.dataSet.validateLabel),axis=0)
        self.model.fit(trainData,trainLabel)

        t1=time.time()
        score=metrics.accuracy_score(self.dataSet.validateLabel,self.model.predict(self.dataSet.validateX))
        cm=metrics.confusion_matrix(self.dataSet.validateLabel,self.model.predict(self.dataSet.validateX))
        print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

def testWinRankClassification(tasktype,queue):
    data=TopcoderWin(tasktype,testratio=0.1,validateratio=0.1)

    data.loadData()

    data.WinRankData()
    data.trainX,data.trainLabel=data.ReSampling(data.trainX,data.trainLabel)
    model=TraditionalClassifier()
    model.dataSet=data
    model.name=data.tasktype+"-classifier(Rank)"
    model.trainModel()
    model.saveModel()
    model.loadModel()
    Y_predict2=model.predict(data.testX)
    print("test score=%f"%(metrics.accuracy_score(data.testLabel,Y_predict2)))
    print("Confusion matrix ")
    print(metrics.confusion_matrix(data.testLabel,Y_predict2))
    kacc=[data.tasktype]
    for k in (3,5,10,20):
        acc=topKAccuracy(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)
        kacc=kacc+[acc]
    print()
    queue.put(kacc)

#test the performance
if __name__ == '__main__':

    with open("../data/TaskInstances/TaskIndex.data","rb") as f:
        tasktypes=pickle.load(f)
    mode=1
    queue=multiprocessing.Queue()
    pool_processes=[]
    for t in tasktypes:
        #if t=="Architecture":continue

        #testWinRankClassification(t)

        p=multiprocessing.Process(target=testWinRankClassification,args=(t,queue))
        pool_processes.append(p)
        p.start()
        #p.join()
    for p in pool_processes:
        p.join()

    result=""
    while queue.empty()==False:
        data=queue.get()
        result=result+data[0]+" : %f"%data[1]
    with open("../data/runResults/rankPrediction.txt","w") as f:
        f.writelines(result)

