import _pickle as pickle
import multiprocessing
from multiprocessing import Condition,Queue
from scipy import sparse
import time
import numpy as np
from DataPrepare.DataContainer import Tasks,UserHistoryGenerator
from Utility import SelectedTaskTypes
from CompetitionGraph.UserRank import rankOnDIG
from Utility.TagsDef import getUsers
#datastructure for reg and sub

class DataURS:

    def __init__(self,tasktype,mode):
        self.userData=UserHistoryGenerator(True).loadActiveUserHistory(tasktype,mode)
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
        return getUsers(self.tasktype+"-test",mode)

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
    def __init__(self,taskid,date,dataset,users,queue,finishSig,scoretag=False):
        multiprocessing.Process.__init__(self)
        self.taskid=taskid
        self.dataset=dataset
        self.users = users
        self.queue=queue
        self.finishSig=finishSig
        if scoretag:
            self.method=self.ScoreBasedMeasure
        else:
            self.method=self.SubNumbasedMeasure

        self.dataset.setTimeline(date)
        self.n_users=len(users)
        self.user_m=sparse.dok_matrix((self.n_users,self.n_users),dtype=np.float32)

    def ScoreBasedMeasure(self,index):
        n_users=self.n_users
        # user a : register and submit
        a = self.users[index]

        regtaskA = self.dataset.getRegTasks(a)
        subtaskA = self.dataset.getSubTasks(a)
        #print("test",regtaskA[:5],subtaskA[:5])
        subscoreA=np.array([])
        if len(subtaskA)>0:
            subscoreA = subtaskA[3]
            subtaskA=subtaskA[0]

        for j in range(index+1,n_users):

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
            com_a=max(1,com_a)
            com_b=max(1,com_b)
            # total avg submit times
            score_a = max(np.sum(subscoreA)/len(regtaskA),1)
            score_b = max(np.sum(subscoreB)/len(regtaskB),1)

            # set entry as outperform degree
            self.user_m[index,j]=(com_a - score_a) / score_a
            self.user_m[j,index]=(com_b - score_b) / score_b

    def SubNumbasedMeasure(self,index):

        n_users=self.n_users
        # user a : register and submit
        a = self.users[index]

        regtaskA = self.dataset.getRegTasks(a)
        subtaskA = self.dataset.getSubTasks(a)
        #print("test",regtaskA[:5],subtaskA[:5])
        subnumA=np.array([])
        if len(subtaskA)>0:
            subnumA = subtaskA[1]
            subtaskA=subtaskA[0]

        for j in range(index+1,n_users):

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
            com_a=max(1,com_a)
            com_b=max(1,com_b)
            # total avg submit times
            sub_a = max(np.sum(subnumA)/len(regtaskA),1)
            sub_b = max(np.sum(subnumB)/len(regtaskB),1)

            # set entry as outperform degree
            self.user_m[index,j]=(com_a - sub_a) / sub_a
            self.user_m[j,index]=(com_b - sub_b) / sub_b

    def run(self):
        for index in range(self.n_users):
            #print(index+1,"of",self.n_users)
            self.method(index)

        #print(self.taskid,"finished")
        self.finishSig.acquire()
        self.queue.put((self.taskid,self.user_m))

        self.finishSig.notify()
        self.finishSig.release()

if __name__ == '__main__':

    mode=2
    scoreTag=False
    tasktypes=SelectedTaskTypes.loadTaskTypes()

    for t in tasktypes["keeped"]:
        if "Code" in t or "Assembly Competition" in t or "First2Finish" in t:continue

        dataset=DataURS(t,mode)
        taskData=Tasks(t)
        taskData.ClipRatio(0.2)
        dataGraph={}
        data={}

        users=list(dataset.getRegUsers())
        print("builiding DIG,users=%d, challenges=%d"%(len(users),len(taskData.taskIDs)))
        #print()
        t0=time.time()
        taskids,postingdate=taskData.taskIDs,taskData.postingdate
        #print(postingdate[:30]); exit(10)
        pool_processes=[]
        max_process_num=10
        queue=Queue()
        finishSig=Condition()
        i=0
        while i < len(taskids):

            date=postingdate[i]

            if max_process_num>len(pool_processes):
                p=UserInteraction(taskid=taskids[i],date=date,dataset=dataset,users=users,
                                  queue=queue,finishSig=finishSig,scoretag=scoreTag)
                p.start()
                pool_processes.append(p)
                i+=1

            else:
                finishSig.acquire()
                #print("pool full")
                if queue.empty()==True:
                    finishSig.wait()

                rmPs=[]
                while queue.empty()==False:
                    result=queue.get()
                    user_m=result[1].toarray()
                    taskid=result[0]
                    data={}
                    data["users"]=users
                    data["data"]=user_m
                    ranks,names=rankOnDIG(data)
                    data={"users":names,"ranks":ranks}
                    #print(ranks)
                    #print(names)
                    #exit(10)
                    #print("fetched",taskid)
                    dataGraph[taskid]=data

                    if len(dataGraph)%100==0:
                        print(len(dataGraph),"of",len(taskids))

                    for j in range(len(pool_processes)):
                        p=pool_processes[j]
                        if p.taskid==taskid:
                            rmPs.append(j)
                            #print(pool_processes[j].taskid,"finished")
                            break

                rmPs.sort()
                rmPs.reverse()
                #print(rmPs)
                for j in rmPs:
                    #pool_processes[j].join()
                    del pool_processes[j]

                finishSig.release()

        while len(pool_processes)>0:

            rmPs=[]
            while queue.empty()==False:
                result=queue.get()
                user_m=result[1].toarray()
                taskid=result[0]
                data={}
                data["users"]=users
                data["data"]=user_m
                ranks,names=rankOnDIG(data)
                data={"users":names,"ranks":ranks}
                #print("fetched",taskid)
                dataGraph[taskid]=data

                if len(dataGraph)%100==0:
                    print(len(dataGraph),"of",len(taskids))

                for i in range(len(pool_processes)):
                    p=pool_processes[i]
                    if p.taskid==taskid:
                        rmPs.append(i)
                        #print(pool_processes[i].taskid,"finished")
                        break

            rmPs.sort()
            rmPs.reverse()
            #print(rmPs)
            for i in rmPs:
                #pool_processes[i].join()
                del pool_processes[i]

        with open("../data/UserInstances/UserGraph/SubNumBased/"+t+"-UserInteraction.data","wb") as f:
            pickle.dump(dataGraph,f,True)

        print("time=%ds"%(time.time()-t0))
        print()
