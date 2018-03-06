import json
import multiprocessing
from DataPrepare.ConnectDB import *
from multiprocessing import Condition,Queue,Pipe
from scipy import sparse
import time
import gc
def loadclusters(choice):
    f=open("../data/clusterResult/clusters"+str(choice)+".json")
    clusters=json.load(f)
    return clusters

#datastructure for reg and sub
class DataURS:
    def __init__(self):
        warnings.filterwarnings("ignore")

        conn = ConnectDB()
        cur = conn.cursor()
        #load registration data
        sqlcmd = "select handle,taskid,regdate from registration"
        cur.execute(sqlcmd)
        self.regdata = np.array(cur.fetchall())

        #load submission data
        sqlcmd="select handle,taskid,subnum,score from submission"
        cur.execute(sqlcmd)
        self.subdata=np.array(cur.fetchall())
        #print(type(self.regdata),self.regdata[:10],self.subdata[:10])
        print("regdata size=%d, subdata size=%d"%(len(self.regdata),len(self.subdata)))

    def setActiveCluster(self,taskIDs):
        self.activeTask=taskIDs
        self.activeReg=[]
        for taskid in taskIDs:
            i=np.where(self.regdata[:,1]==taskid)[0]
            if len(i)>0:
                regs=self.regdata[i]
                for reg in regs:
                    self.activeReg.append(reg)
        self.activeReg=np.array(self.activeReg)
        self.activeSub=[]
        for taskid in taskIDs:
            i=np.where(self.subdata[:,1]==taskid)[0]
            if len(i)>0:
                subs=self.subdata[i]
                for sub in subs:
                    self.activeSub.append(sub)
        self.activeSub=np.array(self.activeSub)
        print("ative regtasks size=%d,active subtasks size=%d"%(len(self.activeReg),len(self.activeSub)))
        #print(self.activeSub[:3],self.activeReg[:3])

    def getRegUsers(self):
        users = set()
        for reg in self.activeReg:
            users.add(reg[0])

        return users

    def getRegTasks(self,user):
        if len(self.activeReg)==0:
            return np.array([])
        indices=np.where(self.activeReg[:,0]==user)[0]
        if len(indices)>0:
            tasks=self.activeReg[indices][:,1]
        else:
            tasks=np.array([])
        return tasks

    def getSubTasks(self,user):
        if len(self.activeSub)==0:
            return np.array([])
        indices=np.where(self.activeSub[:,0]==user)[0]
        if len(indices)>0:
            tasks=self.activeSub[indices][:,1:4]
        else:
            tasks=np.array([])
        return tasks


class UserInteraction(multiprocessing.Process):
    def __init__(self,index,outercon,dataset,users,queue,fsig):
        multiprocessing.Process.__init__(self)
        self.index=index
        self.outercon=outercon
        self.dataset=dataset
        self.users = users
        self.queue = queue
        self.finishedSig=fsig

    def run(self):
        print("process running for user no %d"%self.index)
        n_users=len(self.users)
        userVec=[]
        # user a : register and submit
        a = self.users[self.index]

        regtaskA = self.dataset.getRegTasks(a)
        subtaskA = self.dataset.getSubTasks(a)
        #print("test",regtaskA[:5],subtaskA[:5])
        subnumA=np.array([])
        if len(subtaskA)>0:
            subnumA = subtaskA[:,1]
            subtaskA=subtaskA[:,0]

        t0 = time.time()
        for j in range(self.index+1,n_users):
            if (j + 1) % 5000 == 0:
                print("sub progress: %d/%d for user no %d,cost %ds" % (j + 1, n_users,self.index,time.time()-t0))
                t0=time.time()

            # user b: register and submit
            b = self.users[j]

            regtaskB = dataset.getRegTasks(b)
            subtaskB = dataset.getSubTasks(b)
            subnumB=np.array([])
            if len(subtaskB)>0:
                subnumB = subtaskB[:,1]
                subtaskB=subtaskB[:,0]

            # use common task to compute init status
            comtasks = set(regtaskA).intersection(regtaskB)

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
        #print("put vec index %d"%self.index)
        self.outercon.acquire()
        print("process finished for user no %d"%self.index)

        self.finishedSig.put(self.index)
        self.outercon.notify()
        self.outercon.release()

def constructGraph(tasks,dataset):

    print("cluster size=%d"%len(tasks))
    users=dataset.getRegUsers()
    #print(users)
    queue=Queue()
    users=list(users)
    n_users=len(users)
    userMatrix=sparse.dok_matrix((n_users,n_users))
    print("builiding user matrix,size=%d"%n_users)
    #statistics for user competition status
    cond=Condition()

    maxProcess=min(n_users,20)
    print("running using %d process(es)"%maxProcess)
    pools_process=[]
    finishedSig=Queue()
    cond.acquire()
    for i in range(n_users):
        if len(pools_process)<maxProcess:
            t=UserInteraction(i,cond,dataset,users,queue,finishedSig)
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
                    t.start()

                    break

    cond.release()
    print("gather final data")
    for t in pools_process:
        t.join()

    while queue.empty()==False:
        entries=queue.get()
        for data in entries:
            userMatrix[data[0],data[1]]=data[2]

    return (userMatrix.toarray(),users)

if __name__ == '__main__':
    choice=eval(input("choice= "))
    print("loading data")
    clusters=loadclusters(choice)
    dataset=DataURS()

    for k in clusters.keys():
        if eval(k)<10:
            continue
        print("cluster",k,"graph building")
        cluster=clusters[k]
        #print(cluster)
        cluster=np.array(cluster,dtype=np.int)
        dataset.setActiveCluster(cluster)
        user_m,users=constructGraph(cluster,dataset)
        with open("../data/UserGraph/initGraph/initG_"+str(choice)+"_"+str(k)+".json","w") as f:
            data={}
            data["size"]=len(user_m)
            data["users"]=users
            data["data"]=user_m.tolist()
            json.dump(data,f,ensure_ascii=False)
        gc.collect()
