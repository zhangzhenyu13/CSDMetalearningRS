import json,pickle
import multiprocessing
from DataPrepare.ConnectDB import *
from multiprocessing import Condition,Queue
from scipy import sparse
import time
import gc
import numpy as np
from DataPrepare.DataContainer import Tasks,UserHistoryGenerator
#datastructure for reg and sub

class DataURS:

    def __init__(self,tasktype,mode):
        self.userData=UserHistoryGenerator().loadActiveUserHistory(tasktype,mode)
        self.mode=mode
        self.tasktype=tasktype
        for name in self.userData.keys():
            regtasks=self.userData[name]["regtasks"]
            subtasks=self.userData[name]["subtasks"]
            for i in range(len(regtasks)):
                regtasks[i]=np.array(regtasks[i])
            for i in range(len(subtasks)):
                subtasks[i]=np.array(subtasks[i])
            self.userData[name]["regtasks"]=regtasks
            self.userData[name]["subtasks"]=subtasks

    def getRegUsers(self):
        return self.userData.keys()

    def setTimeline(self,date):
        for name in self.userData.keys():
            #reset regtasks
            regtasks=self.userData[name]["regtasks"]
            #print(regtasks.shape,regtasks[:3])
            indices=np.where(regtasks[1]>date)[0]
            for i in range(len(regtasks)):
                regtasks[i]=regtasks[i][indices]

            self.userData[name]["regtasks"]=regtasks
            #reset subtasks
            subtasks=self.userData[name]["subtasks"]
            #print(subtasks.shape,subtasks[:3])
            indices=np.where(subtasks[2]>date)[0]
            for i in range(len(subtasks)):
                subtasks[i]=subtasks[i][indices]

            self.userData[name]["subtasks"]=subtasks

    def getRegTasks(self,username):
        regtasks=self.userData[username]["regtasks"]

        return regtasks[0]

    def getSubTasks(self,username):
        subtasks=self.userData[username]["subtasks"]

        return subtasks

class UserInteraction(multiprocessing.Process):
    def __init__(self,index,outercon,dataset,users,queue,fsig):
        multiprocessing.Process.__init__(self)
        self.index=index
        self.outercon=outercon
        self.dataset=dataset
        self.users = users
        self.queue = queue
        self.finishedSig=fsig
        self.method=None
    def ScoreBasedMeasure(self):
        n_users=len(self.users)
        userVec=[]
        # user a : register and submit
        a = self.users[self.index]

        regtaskA = self.dataset.getRegTasks(a)
        subtaskA = self.dataset.getSubTasks(a)
        #print("test",regtaskA[:5],subtaskA[:5])
        subscoreA=np.array([])
        if len(subtaskA)>0:
            subscoreA = subtaskA[3]
            subtaskA=subtaskA[0]

        t0 = time.time()
        for j in range(self.index+1,n_users):
            if (j + 1) % 5000 == 0:
                print("sub progress: %d/%d for user no %d,cost %ds" % (j + 1, n_users,self.index,time.time()-t0))
                t0=time.time()

            # user b: register and submit
            b = self.users[j]

            regtaskB = self.dataset.getRegTasks(b)
            subtaskB = self.dataset.getSubTasks(b)
            subscoreB=np.array([])
            if len(subtaskB)>0:
                subscoreB = subtaskB[3]
                subtaskB=subtaskB[0]

            # use common task to compute init status
            comtasks = set(regtaskA).intersection(set(regtaskB))

            if len(comtasks) == 0 or len(subtaskA) == 0 or len(subtaskB) == 0:
                # no interaction, set entry as 0
                continue

            # common avg submit times
            com_a = 0
            com_b = 0

            for taskid in comtasks:
                i=np.where(subtaskA==taskid)[0]
                if len(i)>0:
                    com_a += subscoreA[i[0]]
                i=np.where(subtaskB==taskid)[0]
                if len(i)>0:
                    com_b+=subscoreB[i[0]]

            com_a /= len(comtasks)
            com_b /= len(comtasks)

            # total avg submit times
            score_a = np.sum(subscoreA)/len(regtaskA)
            score_b = np.sum(subscoreB)/len(regtaskB)

            # set entry as outperform degree
            userVec.append((self.index,j,(com_a - score_a) / score_a))
            userVec.append((j,self.index,(com_b - score_b) / score_b))

        self.queue.put(userVec)


    def SubNumbasedMeasure(self):

        n_users=len(self.users)
        userVec=[]
        # user a : register and submit
        a = self.users[self.index]

        regtaskA = self.dataset.getRegTasks(a)
        subtaskA = self.dataset.getSubTasks(a)
        #print("test",regtaskA[:5],subtaskA[:5])
        subnumA=np.array([])
        if len(subtaskA)>0:
            subnumA = subtaskA[1]
            subtaskA=subtaskA[0]

        t0 = time.time()
        for j in range(self.index+1,n_users):
            if (j + 1) % 5000 == 0:
                print("sub progress: %d/%d for user no %d,cost %ds" % (j + 1, n_users,self.index,time.time()-t0))
                t0=time.time()

            # user b: register and submit
            b = self.users[j]

            regtaskB = self.dataset.getRegTasks(b)
            subtaskB = self.dataset.getSubTasks(b)
            subnumB=np.array([])
            if len(subtaskB)>0:
                subnumB = subtaskB[1]
                subtaskB=subtaskB[0]

            # use common task to compute init status
            comtasks = set(regtaskA).intersection(set(regtaskB))

            if len(comtasks) == 0 or len(subtaskA) == 0 or len(subtaskB) == 0:
                # no interaction, set entry as 0
                continue

            # common avg submit times
            com_a = 0
            com_b = 0

            for taskid in comtasks:
                i=np.where(subtaskA==taskid)[0]
                if len(i)>0:
                    com_a += subnumA[i[0]]
                i=np.where(subtaskB==taskid)[0]
                if len(i)>0:
                    com_b+=subnumB[i[0]]

            com_a /= len(comtasks)
            com_b /= len(comtasks)

            # total avg submit times
            sub_a = np.sum(subnumA)/len(regtaskA)
            sub_b = np.sum(subnumB)/len(regtaskB)

            # set entry as outperform degree
            userVec.append((self.index,j,(com_a - sub_a) / sub_a))
            userVec.append((j,self.index,(com_b - sub_b) / sub_b))
            #print("Interaction", a, b,(com_a - sub_a) / sub_a,(com_b - sub_b) / sub_b)

        self.queue.put(userVec)

    def run(self):
        #print("process running for user no %d"%self.index)
        self.method()

        #print("put vec index %d"%self.index)
        self.outercon.acquire()
        #print("process finished for user no %d"%self.index)

        self.finishedSig.put(self.index)
        self.outercon.notify()
        self.outercon.release()

def constructGraph(users,dataset):

    #print(users)
    queue=Queue()

    n_users=len(users)
    userMatrix=sparse.dok_matrix((n_users,n_users))

    #statistics for user competition status
    cond=Condition()

    maxProcess=min(n_users,30)
    #print("running using maximum %d process(es)"%maxProcess)
    pools_process=[]
    finishedSig=Queue()
    cond.acquire()
    for i in range(n_users):
        if len(pools_process)<maxProcess:
            t=UserInteraction(i,cond,dataset,users,queue,finishedSig)
            t.method=t.ScoreBasedMeasure
            t.start()

            pools_process.append(t)
            #print("next user no %d" % (i + 1))
        else:
            if finishedSig.empty():

                cond.notify()
                cond.wait()

            index=finishedSig.get()
            #print("destroy index %d"%index)
            for j in range(len(pools_process)):
                t=pools_process[j]
                #print("checking index of %d"%t.index)
                if t.index==index:
                    #print("clearing data of cache queue")
                    while queue.empty() == False:
                        entries = queue.get()
                        for data in entries:
                            userMatrix[data[0], data[1]] = data[2]
                    t.join()
                    pools_process[j] = UserInteraction(i,cond, dataset, users, queue, finishedSig)
                    t=pools_process[j]
                    t.method=t.ScoreBasedMeasure
                    t.start()

                    break

    cond.release()
    #print("gather final data")
    for t in pools_process:
        t.join()

    while queue.empty()==False:
        entries=queue.get()
        for data in entries:
            userMatrix[data[0],data[1]]=data[2]

    return (userMatrix.toarray(),users)

def initLocalGraph(mode):

    with open("../data/TaskInstances/TaskIndex.data","rb") as f:
        tasktypes=pickle.load(f)

    for t in tasktypes:
        dataset=DataURS(t,mode)
        taskData=Tasks(t,600)
        dataGraph={}
        gc.collect()

        users=list(dataset.getRegUsers())
        print("builiding user matrix,size=%d"%len(users))
        print()
        t0=time.time()
        taskids,postingdate=taskData.taskIDs,taskData.postingdate
        #print(postingdate[:30]); exit(10)
        for i in range(len(taskids)):
            if (i+1)%30==0:
                print(i+1,"time=%ds"%(time.time()-t0))
                t0=time.time()

            date=postingdate[i]
            dataset.setTimeline(date)

            user_m,users=constructGraph(users,dataset)

            data={}
            data["size"]=len(user_m)
            data["users"]=users
            data["data"]=user_m
            dataGraph[taskids[i]]=data

        with open("../data/UserInstances/UserGraph/ScoreBased/"+t+"-UserInteraction.data","wb") as f:
            pickle.dump(dataGraph,f)


if __name__ == '__main__':
    mode=1
    initLocalGraph(mode)
