from ML_Models.CascadingModel import *
from DataPrepare.TopcoderDataSet import TopcoderWin
import multiprocessing
def loadTestData(tasktype):
    data=TopcoderWin(tasktype,testratio=1,validateratio=0)
    data.setParameter(tasktype,2,True)
    data.loadData()
    data.WinClassificationData()
    return data

#cascading models
def testCascadingModel():
    model=CascadingModel(tasktype=tasktype)
    taskids=data.taskids[:data.testPoint]
    mymetric=model.mymetric
    mymetric.callall=False
    print("\n meta-learning model top k acc")
    Y_predict2=model.predict(data.testX,taskids)
    for k in (3,5,10):
        acc=mymetric.topKPossibleUsers(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKRUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        print()
    #exit(10)
    print()
    mymetric.verbose=0
    print("tuning cascading models")
    for k in(3,5,10):
        model.topK=k
        model.metaWin=WinnerSel[k]
        model.fit(data.testX)
        model.saveConf()

    mymetric.verbose=1
    model.setVerbose(1)

    print("\n meta-learning model top k acc")
    for k in (3,5,10):
        model.topK=k
        model.loadConf()
        model.loadModel()
        Y_predict2=model.predict(data.testX,taskids)
        #metrics
        acc=mymetric.topKPossibleUsers(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKRUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,data,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        print()

    #mymetric.verbose=0
    #params=bestPDIG(mymetric,Y_predict2,data)
    #print(params)

    #exit(10)

def testBestReRank():
    data=TopcoderWin(tasktype,testratio=1,validateratio=0)
    data.setParameter(tasktype,2,True)
    data.loadData()
    data.WinClassificationData()
    mymetric=model.mymetric
    mymetric.verbose=0
    taskids=data.taskids[:data.testPoint]

    print("\n meta-learning model top k acc")
    for k in (3,5,10):
        model.topK=k
        model.loadConf()
        model.loadModel()
        Y_predict2=model.predict(data.testX,taskids)
        #metrics
        acc=mymetric.topKPossibleUsers(Y_predict2,data,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKPDIGUsers(Y_predict2,data,k,1)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        print()

    print("tuning re-rank weight")
    best_param={3:0,5:0,10:0}

    for k in (3,5,10):
        model.topK=k
        maxAcc=[0,0]
        model.loadConf()
        model.loadModel()
        Y_predict2=model.predict(data.testX,taskids)

        for w in range(0,11):
            acc=mymetric.topKPDIGUsers(Y_predict2,data,k,w/10)
            acc=np.mean(acc)
            if acc>maxAcc[0]:
                maxAcc=[acc,w/10]
            print(data.tasktype,"top %d"%k,acc,"weight=%f"%(w/10))
        best_param[k]=maxAcc[1]
        print(data.tasktype,"top %d"%k,maxAcc[0],"weight=%f"%maxAcc[1])
        print()

    return best_param


#test winning
class TuneTask(multiprocessing.Process):
    maxProcessNum=28
    def __init__(self,tuneID,params,data,queue,cond):
        multiprocessing.Process.__init__(self)
        self.tuneID=tuneID
        self.model=CascadingModel(**params)
        self.data=data
        self.queue=queue
        self.params=params
        self.cond=cond

    def run(self):

        bestScore=0
        topDig=[w/10 for w in range(0,11)]
        topDig.reverse()
        tD=0
        for self.model.topDig in topDig:
            acc=self.model.score(self.data)
            if acc>bestScore:
                tD=self.model.topDig
                bestScore=acc

        self.model.topDig=tD
        self.params["topDig"]=self.model.topDig
        self.cond.acquire()
        self.queue.put([self.tuneID,self.model,bestScore])
        print("%4.3f"%bestScore,"running ID: %d"%self.tuneID,"finished Tuning topDig",self.params)
        self.cond.notify()
        self.cond.release()


def TuneBestPara():
    params={"regThreshold":1,
            "subThreshold":1,
            "topDig":1,
            "metaReg":1,"metaSub":1,"metaWin":1,
            "topK":3,"tasktype":tasktype
        }

    regT=[w/10 for w in range(0,11)]
    subT=[w/10 for w in range(0,11)]
    queue=multiprocessing.Queue()
    cond=multiprocessing.Condition()

    for topK in range(3,5,10):
        bestModel=None
        bestScore=0
        pools={}
        params["metaWin"]=WinnerSel[topK]
        params["topK"]=topK
        tuneID=0
        print("searching for top%d\n"%topK)
        for metaReg in (1,2,3):
            params["metaReg"]=metaReg

            for metaSub in (1,2,3):
                params["metaSub"]=metaSub

                for regThreshold in regT:
                    params["regThreshold"]=regThreshold

                    for subThreshold in subT:
                        params["subThreshold"]=subThreshold

                        if len(pools)<TuneTask.maxProcessNum:

                            p=TuneTask(tuneID,params,data,queue,cond)
                            p.start()
                            pools[tuneID]=p
                            tuneID+=1
                        else:
                            cond.acquire()
                            if queue.empty():
                                cond.wait()
                            while queue.empty()==False:
                                entry=queue.get()
                                if entry[2]>bestScore:
                                    bestModel=entry[1]
                                    bestScore=entry[2]
                                p=pools[entry[0]]
                                #p.join()
                                del pools[entry[0]]

                            p=TuneTask(tuneID,params,data,queue,cond)
                            p.start()
                            pools[tuneID]=p
                            tuneID+=1
                            cond.release()
        print("==============================>")
        print("gather final data...\n")
        while queue.empty()==False:
            entry=queue.get()
            if entry[2]>bestScore:
                bestModel=entry[1]
                bestScore=entry[2]
            p=pools[entry[0]]
            #p.join()
            del pools[entry[0]]

        #model.saveConf()
        print()
        print("top%d"%topK,"acc=%f"%bestScore)
        print()

if __name__ == '__main__':

    '''
    ml_model={
        1:DNNCLassifier,
        2:EnsembleClassifier,
        3:XGBoostClassifier
    }
    '''

    WinnerSel={
        3:1,
        5:1,
        10:1
    }

    tasktype="Architecture"
    data=loadTestData(tasktype)

    TuneBestPara()
    testBestReRank()
