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

    print()

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
    data=loadTestData(tasktype)
    model=CascadingModel(tasktype)
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
    maxProcessNum=30
    def __init__(self,tuneID,params,data,queue):
        multiprocessing.Process.__init__(self)
        self.tuneID=tuneID
        self.model=CascadingModel(**params)
        self.data=data
        self.queue=queue
        self.params=params

    def run(self):

        bestScore=0
        topDig=[w/10 for w in range(0,11)]
        topDig.reverse()
        tD=0
        #print("begin,%d"%self.tuneID)
        mymetric=self.model.mymetric


        for self.model.topDig in topDig:
            #print("predicting")
            Y=self.model.predict(data.testX,data.taskids[:self.data.testPoint])
            #print("predict=>topk")
            acc=self.model.score(self.data)
            #print(acc,"topDig=%f"%self.model.topDig)
            if acc>bestScore:
                tD=self.model.topDig
                bestScore=acc

        self.model.topDig=tD
        self.params["topDig"]=self.model.topDig

        #print("finished")
        #self.cond.acquire()
        self.queue.put([self.tuneID,self.model,bestScore,self.params])
        #print(bestScore,self.tuneID)
        #self.cond.notify()
        #self.cond.release()


def TuneBestPara():

    params={"regThreshold":1,
            "subThreshold":1,
            "topDig":1,
            "metaReg":1,"metaSub":1,"metaWin":1,
            "topK":3,"tasktype":tasktype
        }

    regT=[w/10 for w in range(0,11)]
    subT=[w/10 for w in range(0,11)]
    metaRs=(1,2,3)
    metaSs=(1,2,3)
    metaWs=(1,2,3)

    queue=multiprocessing.Queue(TuneTask.maxProcessNum)

    for topK in range(3,5,10):
        bestModel=None
        bestScore=0
        pools={}
        #params["metaWin"]=WinnerSel[topK]
        params["topK"]=topK
        tuneID=0
        #processes_pool=multiprocessing.Pool(processes=TuneTask.maxProcessNum)
        print("searching for top%d\n"%topK)
        progress=1
        for metaReg in metaRs:
            params["metaReg"]=metaReg

            for metaSub in metaSs:
                params["metaSub"]=metaSub

                for metaWin in metaWs:
                    params["metaWin"]=metaWin
                    print("progress=%d/27"%progress)
                    progress+=1

                    for regThreshold in regT:
                        params["regThreshold"]=regThreshold

                        for subThreshold in subT:
                            params["subThreshold"]=subThreshold
                            #if tuneID<72:tuneID+=1;params["verbose"]=2;continue

                            if len(pools)<TuneTask.maxProcessNum:
                                #print("not full,size=%d"%len(pools),tuneID)

                                p=TuneTask(tuneID,params,data,queue)
                                p.start()
                                pools[tuneID]=p
                                tuneID+=1

                            else:

                                #cond.acquire()
                                #print("full pool size=%d"%len(pools))

                                entry=queue.get(block=True)
                                if entry[2]>bestScore:
                                    bestModel=entry[1]
                                    bestScore=entry[2]

                                    print("%4.3f"%bestScore,"ID:%d"%entry[0],entry[3])

                                p=pools[entry[0]]
                                p.join()
                                #print(pools.keys(),"del=>",entry[0])
                                del pools[entry[0]]

                                while queue.qsize()>0:
                                    entry=queue.get()
                                    if entry[2]>bestScore:
                                        bestModel=entry[1]
                                        bestScore=entry[2]

                                        print("%4.3f"%bestScore,"ID:%d"%entry[0],entry[3])

                                    p=pools[entry[0]]
                                    p.join()
                                    #print(pools.keys(),"del=>",entry[0])
                                    del pools[entry[0]]

                                #print("del pool size=%d"%len(pools))

                                p=TuneTask(tuneID,params,data,queue)
                                p.start()
                                pools[tuneID]=p
                                tuneID+=1

                                #cond.release()

        print("==============================>")
        print("gather final data...\n")
        while queue.empty()==False:
            entry=queue.get()
            if entry[2]>bestScore:
                bestModel=entry[1]
                bestScore=entry[2]
            p=pools[entry[0]]
            p.join()
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

    WinnerSels={
        3:1,
        5:1,
        10:1
    }

    tasktype="Test Suites"
    data=loadTestData(tasktype)

    TuneBestPara()
    #testBestReRank()
