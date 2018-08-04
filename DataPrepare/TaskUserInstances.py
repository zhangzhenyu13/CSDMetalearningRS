import multiprocessing
import time,gc
from DataPrepare.DataContainer import *
from Utility.TagsDef import *
from Utility import SelectedTaskTypes
import _pickle as pickle

class DataInstances(multiprocessing.Process):
    maxProcessNum=25

    def __init__(self,tasktype,cond,queue,usingmode,testMode=False):
        multiprocessing.Process.__init__(self)
        self.tasktype=tasktype
        if testMode:
            self.tasktype=tasktype+"-test"
        self.testMode=testMode
        self.cond=cond
        self.queue=queue
        self.usingMode=usingmode
        self.running=True

        self.filePath="../data/TopcoderDataSet/"+ModeTag[self.usingMode].lower()+\
                 "HistoryBasedData/"+self.tasktype+"-user_task.data"


    def run(self):
        self.loadData()

        self.watchUsers()

        self.createInstancesWithHistoryInfo()

        print(self.tasktype+"=>finished running")
        with open("../data/runResults/genTrainData"+ModeTag[self.usingMode],"a") as f:
            f.writelines(self.tasktype+"\n")

        self.cond.acquire()
        self.queue.put(self.tasktype)
        self.cond.notify()
        self.cond.release()

    def loadData(self):
        #load task data
        self.selTasks=Tasks(tasktype=self.tasktype.replace("-test",""))
        if self.testMode:
            self.selTasks.ClipRatio(0.2)
        #print("posting date",self.selTasks.postingdate[:20])
        #load user data
        self.userdata=UserHistoryGenerator()

        #load reg data
        with open("../data/TaskInstances/RegInfo/"+self.tasktype.replace("-test","")+"-regs.data","rb") as f:
            data=pickle.load(f)
            for k in data.keys():
                data[k]=data[k].tolist()
                data[k].reverse()
            ids=data["taskids"]
            dates=data["regdates"]
            #print(dates[:30])
            names=data["names"]
            self.regdata=RegistrationDataContainer(tasktype=self.tasktype,taskids=ids,usernames=names,regdates=dates)
            print("loaded %d reg items"%len(self.regdata.taskids))

        #load sub data
        with open("../data/TaskInstances/SubInfo/"+self.tasktype.replace("-test","")+"-subs.data","rb") as f:
            data=pickle.load(f)
            ids=data["taskids"]
            dates=data["subdates"]
            #print(dates[:30]);exit(10)
            names=data["names"]
            subnums=data["subnums"]
            scores=data["scores"]
            ranks=data["finalranks"]
            self.subdata=SubmissionDataContainer(tasktype=self.tasktype,taskids=ids,usernames=names,
                                                 subnums=subnums,subdates=dates,scores=scores,finalranks=ranks)
            print("loaded %d sub items"%len(self.subdata.taskids))

    def saveDataIndex(self,filepath,dataSegment):
        with open(filepath,"wb") as f:
            data=[]
            for seg in range(dataSegment):
                data.append(filepath+str(seg))
            pickle.dump(data,f)

    def watchUsers(self):

        userData=self.userdata.loadActiveUserHistory(tasktype=self.tasktype,mode=self.usingMode)
        UsersIndex=getUsers(self.tasktype,self.usingMode)

        watchPoint=int(0.7*len(self.selTasks.taskIDs))-1

        for index in range(len(self.selTasks.taskIDs)):


            id,date=self.selTasks.taskIDs[index],self.selTasks.postingdate[index]

            reg_usernams, regDates=self.regdata.getRegUsers(id)

            if reg_usernams is None:
                continue

            naiveUcount=0
            if watchPoint==index:
                for nth in range(len(UsersIndex)):
                    name=UsersIndex[nth]
                    tenure= userData[name]["tenure"]
                    if tenure<=date:
                        naiveUcount+=1
                print("\n %d cold users in %s\n"%(naiveUcount,self.tasktype))
                exit(12)

            for nth in range(len(UsersIndex)):
                name=UsersIndex[nth]

                tenure, skills,skills_vec = \
                    userData[name]["tenure"],userData[name]["skills"],userData[name]["skills_vec"]

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][0] <= date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], 0, axis=0)
                userData[name]["regtasks"] = regtasks

                subtasks = userData[name]["subtasks"]
                while len(subtasks[0]) > 0 and subtasks[2][0] <= date:
                    for l in range(len(subtasks)):
                        subtasks[l] = np.delete(subtasks[l], 0, axis=0)
                userData[name]["subtasks"] = subtasks


    def createInstancesWithHistoryInfo(self,threshold=6e+5,verboseNum=1e+3):
        filepath=self.filePath

        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]

        dataIndex=self.selTasks.dataIndex

        userData=self.userdata.loadActiveUserHistory(tasktype=self.tasktype,mode=self.usingMode)
        UsersIndex=getUsers(self.tasktype,self.usingMode)

        print(self.tasktype+"(mode:%d)=>:"%self.usingMode,"construct instances with %d tasks and %d users" %
              (len(self.selTasks.taskIDs), len(UsersIndex)) )

        missingtask=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.selTasks.taskIDs)):


            if (index+1)%verboseNum==0:
                print(self.tasktype+"=>:",index+1,"of",len(self.selTasks.taskIDs),
                      "current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                t0=time.time()

            id,date=self.selTasks.taskIDs[index],self.selTasks.postingdate[index]

            reg_usernams, regDates=self.regdata.getRegUsers(id)

            if reg_usernams is None:
                missingtask+=1
                continue

            for nth in range(len(UsersIndex)):
                name=UsersIndex[nth]

                tenure, skills,skills_vec = \
                    userData[name]["tenure"],userData[name]["skills"],userData[name]["skills_vec"]

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][0] <= date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], 0, axis=0)
                userData[name]["regtasks"] = regtasks

                subtasks = userData[name]["subtasks"]
                while len(subtasks[0]) > 0 and subtasks[2][0] <= date:
                    for l in range(len(subtasks)):
                        subtasks[l] = np.delete(subtasks[l], 0, axis=0)
                userData[name]["subtasks"] = subtasks

                # reg history info
                if len(regtasks[0])>0:
                    regID, regDate = regtasks[0], regtasks[1]

                    date_interval = regDate[len(regDate)-1] - date
                    participate_recency = regDate[0]-date
                    participate_frequency = len(regID)
                else:
                    date_interval=0
                    participate_recency=1e+6
                    participate_frequency=0

                # sub history info
                if len(subtasks[0])>0:
                    subID, subNum, subDate, subScore, subrank = \
                        subtasks[0], subtasks[1], subtasks[2], subtasks[3], subtasks[4]

                    commit_recency = subDate[0]-date
                    commit_frequency = np.sum(subNum)
                    last_perfromance = subScore[0]
                    last_rank=subrank[0]

                    win_indices = np.where(subrank == 0)[0]
                    win_frequency = len(win_indices)
                    win_recency = 1e+6
                    for i in range(len(subID)):
                        if subrank[i] == 0:
                            win_recency = subDate[i]
                            break
                else:
                    commit_recency=1e+6
                    commit_frequency=0
                    last_perfromance=0
                    last_rank=10
                    win_frequency=0
                    win_recency=1e+6

                user=[tenure-date,date_interval,
                      participate_recency,participate_frequency,
                      commit_recency,commit_frequency,
                      win_recency,win_frequency,
                      last_perfromance,last_rank]
                user=user+list(skills_vec)

                taskPos=dataIndex[id]
                lan,tech,prize,duration,diffdeg=self.selTasks.lans[taskPos],self.selTasks.techs[taskPos],\
                    self.selTasks.prizes[taskPos],self.selTasks.durations[taskPos],self.selTasks.diffdegs[taskPos]

                task=[]
                skills=set(skills)
                if len(lan)==0:
                    task.append(1)
                else:
                    lan=set(lan)
                    task.append(len(lan.intersection(skills))/len(lan))
                if len(tech)==0:
                    task.append(1)
                else:
                    tech=set(tech)
                    task.append(len(tech.intersection(skills))/len(tech))
                task.append(diffdeg)
                task.append(duration)
                task.append(prize)
                task.append(date)

                #task vec
                task=task+list(self.selTasks.docX[taskPos])

                usernames.append(name)
                taskids.append(id)
                users.append(user)
                tasks.append(task)
                dates.append(date)

                if name in reg_usernams:
                    regists.append(1)
                    #performance
                    curPerformance = self.subdata.getResultOfSubmit(name, id)
                    if curPerformance is not None:
                        submits.append(curPerformance[0])
                        ranks.append(curPerformance[1])
                        scores.append(curPerformance[2])
                    else:
                        submits.append(0)
                        ranks.append(10)
                        scores.append(0)
                else:
                    regists.append(0)
                    submits.append(0)
                    ranks.append(10)
                    scores.append(0)

                '''
                if len(userData[name]["regtasks"][1][:20])>10:
                    print()
                    print(date)
                    print("reg",userData[name]["regtasks"][1][:10])
                    print("sub",userData[name]["subtasks"][2][:10])
                    print("task",self.selTasks.postingdate[:10])
                    print("uservec",user)
                    print("task vec",task)
                    exit(10)
                '''

            if len(taskids)>threshold:
                data={}
                #print(self.tasktype+"=>:","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
                data["usernames"] = usernames
                data["taskids"] = taskids
                data["tasks"] = tasks
                data["users"] = users
                data["dates"] = dates
                data["submits"] = submits
                data["ranks"] = ranks
                data["scores"]=scores
                data["regists"]=regists
                with open(filepath+str(dataSegment),"wb") as f:
                    pickle.dump(data,f,True)
                #reset for next segment
                data=None
                tasks=[]
                users=[]
                usernames = []
                taskids = []
                dates = []
                submits = []
                ranks = []
                scores=[]
                regists=[]
                gc.collect()
                dataSegment+=1

        data={}

        data["usernames"] = usernames
        data["taskids"] = taskids
        data["tasks"] = tasks
        data["users"] = users
        data["dates"] = dates
        data["submits"] = submits
        data["ranks"] = ranks
        data["scores"]=scores
        data["regists"]=regists

        #print(self.tasktype+"=>:","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
        with open(filepath+str(dataSegment),"wb") as f:
            pickle.dump(data,f,True)

        self.saveDataIndex(filepath=filepath,dataSegment=dataSegment+1)
        #print(self.tasktype+"=>:","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        #print()

def genDataSet():

    process_pools=[]

    tasktypes=SelectedTaskTypes.loadTaskTypes()

    for t in tasktypes["keeped"]:

        if len(process_pools)<DataInstances.maxProcessNum:
            proc=DataInstances(tasktype=t,cond=cond,queue=queue,usingmode=mode,testMode=testInst)
            proc.start()
            process_pools.append(proc)
        else:
            cond.acquire()
            if queue.empty():
                cond.wait()
            finishP=[]
            while queue.empty()==False:
                finishP.append(queue.get())
            rmIndices=[]
            for i in range(len(process_pools)):
                if process_pools[i].tasktype in finishP:
                    rmIndices.append(i)
                    process_pools[i].join()

            rmIndices.reverse()
            for i in rmIndices:
                del process_pools[i]

            proc=DataInstances(tasktype=t,cond=cond,queue=queue,usingmode=mode,testMode=testInst)
            proc.start()
            process_pools.append(proc)
            cond.release()

if __name__ == '__main__':

    cond=multiprocessing.Condition()
    queue=multiprocessing.Queue()

    mode=2
    testInst=False
    #DataInstances(tasktype="global",cond=cond,queue=queue,usingmode=mode).start();exit()
    genDataSet()

