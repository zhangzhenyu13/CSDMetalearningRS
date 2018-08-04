from ML_Models.CascadingModel import *
from DataPrepare.TopcoderDataSet import TopcoderWin
import multiprocessing

def loadTestData(tasktype,clipratio=1.0,dropratio=0.0):
    data=TopcoderWin(tasktype,testratio=clipratio,validateratio=dropratio)
    data.setParameter(tasktype,2,True)
    data.loadData()
    data.WinClassificationData()
    return data

#cascading models
def testCascadingModel():
    model=CascadingModel(tasktype=modeltype)
    taskids=data.taskids[:data.testPoint]
    Y_label=data.testLabel[:data.testPoint]
    Y_sublabel=data.submitLabelClassification[:data.testPoint]
    Y_reglabel=data.registerLabelClassification[:data.testPoint]
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
        acc=mymetric.topKPossibleUsers(Y_predict2,Y_label,k)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKRUsers(Y_predict2,Y_label,Y_reglabel,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        acc=mymetric.topKSUsers(Y_predict2,Y_label,Y_sublabel,k,)
        acc=np.mean(acc)
        print(data.tasktype,"top %d"%k,acc)

        print()

    #mymetric.verbose=0
    #params=bestPDIG(mymetric,Y_predict2,data)
    #print(params)

    #exit(10)

class TuneDIG(multiprocessing.Process):
    maxProcessNum=32
    def __init__(self,tuneID,model,data,queue):
        multiprocessing.Process.__init__(self)
        self.tuneID=tuneID
        self.model=model
        #if transferLearning:

        self.data=data
        self.queue=queue

    def run(self):

        bestScore=0
        ran_weighte=[w/10 for w in range(10,11)]
        ran_weighte.reverse()
        best_w=0
        #print("begin,%d"%self.tuneID)
        mymetric=self.model.mymetric
        taskids=self.data.taskids[self.data.validatePoint:]
        X=self.data.trainX
        Y_label=self.data.trainLabel
        for rw in ran_weighte:
            #print("predicting")
            Y=self.model.predict(X,taskids)
            Y=mymetric.topKPDIGUsers(Y,Y_label,taskids,self.model.topK,rw)
            acc=np.mean(Y)

            if acc>bestScore:
                best_w=rw
                bestScore=acc

        self.queue.put([self.tuneID,self.model,bestScore,best_w])


def testBestReRank(k,bestModel=None):
    if bestModel is None:
        model=CascadingModel(tasktype=modeltype,digtype=datatype)
    else:
        model=bestModel
    mymetric=model.mymetric
    mymetric.verbose=0
    taskids=data.taskids[:data.testPoint]
    Y_label=data.testLabel[:data.testPoint]

    print("tuning re-rank weight\n")
    best_param={3:0,5:0,10:0}
    rw=[w/10 for w in range(0,11)]
    rw.reverse()


    model.topK=k
    maxAcc=[0,0]
    model.loadConf()
    model.loadModel()
    Y_predict2=model.predict(data.testX,taskids)

    for w in rw:
            acc=mymetric.topKPDIGUsers(Y_predict2,Y_label,taskids,k,w)
            acc=np.mean(acc)
            if acc>maxAcc[0]:
                maxAcc=[acc,w]
                #print(data.tasktype,"top %d"%k,acc,"weight=%f"%(w/10))
    best_param[k]=maxAcc[1]
    print("\n",data.tasktype,"top %d"%k,maxAcc[0],"weight=%f"%maxAcc[1])
    print()

#test winning
class TuneTask(multiprocessing.Process):
    maxProcessNum=32
    def __init__(self,tuneID,params,data,queue):
        multiprocessing.Process.__init__(self)
        self.tuneID=tuneID
        self.model=CascadingModel(**params)
        #if transferLearning:

        self.data=data
        self.queue=queue
        self.params=params

    def run(self):

        bestScore=0
        topDig=[w/10 for w in range(10,11)]
        topDig.reverse()
        tD=0
        #print("begin,%d"%self.tuneID)
        mymetric=self.model.mymetric
        taskids=self.data.taskids[self.data.validatePoint:]
        X=self.data.trainX
        Y_label=self.data.trainLabel
        for self.model.topDig in topDig:
            #print("predicting")
            Y=self.model.predict(X,taskids)
            Y=mymetric.topKPossibleUsers(Y,Y_label,self.model.topK)
            acc=np.mean(Y)
            #acc=self.model.score(self.data)
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


def TuneBestPara(topK):

    params={"regThreshold":1,
            "subThreshold":1,
            "topDig":1,
            "metaReg":1,"metaSub":1,"metaWin":1,
            "topK":topK,"tasktype":modeltype,#,"verbose":2
            "digtype":datatype
        }

    regT=[w/10 for w in range(1,11)]
    subT=[w/10 for w in range(1,11)]
    metaRs=(1,2)
    metaSs=(1,2)
    metaWs=(1,2)
    n_tasks=len(metaRs)*len(metaSs)*len(metaWs)


    print("searching for top%d\n"%topK)

    queue=multiprocessing.Queue(TuneTask.maxProcessNum)

    bestModel=None
    bestScore=0
    pools={}
    #params["metaWin"]=WinnerSel[topK]
    params["topK"]=topK
    tuneID=0
    #processes_pool=multiprocessing.Pool(processes=TuneTask.maxProcessNum)
    progress=1


    for metaReg in metaRs:
            params["metaReg"]=metaReg

            for metaSub in metaSs:
                params["metaSub"]=metaSub

                for metaWin in metaWs:
                    params["metaWin"]=metaWin
                    print("progress=%d/%d"%(progress,n_tasks))
                    progress+=1

                    for regThreshold in regT:
                        params["regThreshold"]=regThreshold

                        for subThreshold in subT:
                            params["subThreshold"]=subThreshold
                            #if tuneID<72:tuneID+=1;params["verbose"]=2;continue
                            if (tuneID+1)%30==0:print("finished %d"%(tuneID+1))

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

    left_n=len(pools)
    for i in range(left_n):
        entry=queue.get()
        if entry[2]>bestScore:
            bestModel=entry[1]
            bestScore=entry[2]
        p=pools[entry[0]]
        p.join()
        del pools[entry[0]]


    queue.close()
    if transferLearning ==False:
        bestModel.saveConf()

    print()
    bestModel.setVerbose(1)
    testAcc=bestModel.score(data)
    print("top%d"%topK,"train acc=%4.3f"%bestScore,"test acc=%4.3f"%testAcc)
    print()
    return bestModel

if __name__ == '__main__':

    clipratio=0.5
    dropratio=0.
    transferLearning=True
    datatype="First2Finish#3"
    modeltype="First2Finish"
    data=loadTestData(datatype,clipratio,dropratio)

    topK=3
    bestModel=TuneBestPara(topK=topK)

    testBestReRank(topK,bestModel)
