import multiprocessing
import time,gc
from DataPrepare.DataContainer import *
from Utility.TagsDef import *
class DataInstances(multiprocessing.Process):
    maxProcessNum=16

    def __init__(self,tasktype,cond,usingmode,clusternum=0):
        multiprocessing.Process.__init__(self)
        self.tasktype=tasktype.replace("/","_")
        self.cond=cond
        self.usingMode=usingmode
        self.clusternum=clusternum
        self.running=True

    def run(self):
        self.loadData()

        if self.usingMode==0:
            self.createInstancesWithRegHistoryInfo()
        elif self.usingMode==1:
            self.createInstancesWithSubHistoryInfo()
        elif self.usingMode==2:
            self.createInstancesWithWinHistoryInfo()
        else:
            print("error mode")
            raise AssertionError()

        print(self.tasktype+"=>finished running")
        with open("../data/runResults/genTrainData"+ModeTag[self.usingMode],"a") as f:
            f.writelines(self.tasktype+"\n")

    def loadData(self):
        #load task data
        self.selTasks=Tasks(tasktype=self.tasktype)
        #print("posting date",self.selTasks.postingdate[:20])
        #load user data
        self.userdata=UserHistoryGenerator()

        #load reg data
        with open("../data/TaskInstances/RegInfo/"+self.tasktype+"-regs.data","rb") as f:
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
        with open("../data/TaskInstances/SubInfo/"+self.tasktype+"-subs.data","rb") as f:
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

    def createInstancesWithRegHistoryInfo(self,threshold=6e+5,verboseNum=1e+5):

        filepath="../data/TopcoderDataSet/regHistoryBasedData/"+self.tasktype+"-user_task.data"

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

        userData=self.userdata.loadActiveUserHistory(tasktype=self.tasktype,mode=0)

        print(self.tasktype+"=>:","construct registration history based instances with %d tasks and %d users" %
              (len(dataIndex), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.selTasks.taskIDs)):
            if (index+1)%verboseNum==0:
                print(self.tasktype+"=>:",index+1,"of",len(self.selTasks.taskIDs),
                      "current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                t0=time.time()

            id,date=self.selTasks.taskIDs[index],self.selTasks.postingdate[index]

            # task data of id

            reg_usernams, regDates=self.regdata.getRegUsers(id)
            if reg_usernams is None:
                missingtask+=1
                continue

            for name in userData.keys():

                tenure, skills,skills_vec = userData[name]["tenure"],userData[name]["skills"],userData[name]["skills_vec"]

                if tenure is None or tenure<date:
                    #no such user in user data
                    missinguser+=1
                    continue

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][0] < date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], 0, axis=0)
                userData[name]["regtasks"] = regtasks

                if len(regtasks[0]) == 0:
                    missinguser += 1
                    continue

                if name in reg_usernams:
                    regists.append(1)
                else:
                    regists.append(0)

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

                #print("reg history of",name,len(regtasks[0]))

                # reg history info
                regID, regDate = regtasks[0], regtasks[1]

                date_interval = regDate[len(regDate)-1] - date
                participate_recency = regDate[0]-date
                participate_frequency = len(regID)

                #user vec
                user=[tenure-date,date_interval,participate_recency,participate_frequency]
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

                #task vec
                task=task+list(self.selTasks.docX[taskPos])

                usernames.append(name)
                taskids.append(id)
                users.append(user)
                tasks.append(task)
                dates.append(date)

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
                    pickle.dump(data,f)
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
            pickle.dump(data,f)

        self.saveDataIndex(filepath=filepath,dataSegment=dataSegment+1)

        #print(self.tasktype+"=>:","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        #print()

    def createInstancesWithSubHistoryInfo(self,threshold=6e+5,verboseNum=1e+5):
        filepath="../data/TopcoderDataSet/subHistoryBasedData/"+self.tasktype+"-user_task.data"

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

        userData=self.userdata.loadActiveUserHistory(mode=1,tasktype=self.tasktype)

        print(self.tasktype+"=>:","construct submission history based instances with %d tasks and %d users" %
              (len(dataIndex), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.selTasks.taskIDs)):
            if (index+1)%verboseNum==0:
                print(self.tasktype+"=>:",index+1,"of",len(self.selTasks.taskIDs),
                      "current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                print()
                t0=time.time()

            id,date=self.selTasks.taskIDs[index],self.selTasks.postingdate[index]

            reg_usernams, regDates=self.regdata.getRegUsers(id)

            if reg_usernams is None:
                missingtask+=1
                continue
            for name in userData.keys():

                tenure, skills,skills_vec = userData[name]["tenure"],userData[name]["skills"],userData[name]["skills_vec"]

                if tenure is None or tenure<date:
                    #no such user in user data
                    missinguser+=1
                    continue

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][0] < date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], 0, axis=0)
                userData[name]["regtasks"] = regtasks

                if len(regtasks[0]) == 0:
                    missinguser += 1
                    continue


                subtasks = userData[name]["subtasks"]
                while len(subtasks[0]) > 0 and subtasks[2][0] < date:
                    for l in range(len(subtasks)):
                        subtasks[l] = np.delete(subtasks[l], 0, axis=0)
                userData[name]["subtasks"] = subtasks

                if len(subtasks[0])==0:
                    missinguser+=1
                    continue

                if name in reg_usernams:
                    regists.append(1)
                else:
                    regists.append(0)
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

                #print("reg and sub history of",name,len(regtasks[0]),len(subtasks[0]))

                # reg history info
                regID, regDate = regtasks[0], regtasks[1]

                date_interval = regDate[len(regDate)-1] - date
                participate_recency = regDate[0]-date
                participate_frequency = len(regID)

                # sub history info
                subID, subNum, subDate, subScore, subrank = subtasks[0], subtasks[1], subtasks[2], subtasks[3], subtasks[4]

                commit_recency = subDate[0]-date
                commit_frequency = np.sum(subNum)
                last_perfromance = subScore[0]
                last_rank=subScore[0]

                user=[tenure-date,date_interval,participate_recency,participate_frequency,commit_recency,commit_frequency,
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

                #task vec
                task=task+list(self.selTasks.docX[taskPos])

                usernames.append(name)
                taskids.append(id)
                users.append(user)
                tasks.append(task)
                dates.append(date)

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
                    pickle.dump(data,f)
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
            pickle.dump(data,f)

        self.saveDataIndex(filepath=filepath,dataSegment=dataSegment+1)
        #print(self.tasktype+"=>:","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        #print()

    def createInstancesWithWinHistoryInfo(self,threshold=6e+5,verboseNum=1e+5):
        filepath="../data/TopcoderDataSet/winHistoryBasedData/"+self.tasktype+"-user_task.data"

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

        userData=self.userdata.loadActiveUserHistory(tasktype=self.tasktype,mode=2)

        print(self.tasktype+"=>:","construct winning history based instances with %d tasks and %d users" %
              (len(dataIndex), len(userData.keys())))

        missingtask=0
        missinguser=0
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
            for name in userData.keys():

                tenure, skills,skills_vec = userData[name]["tenure"],userData[name]["skills"],userData[name]["skills_vec"]

                if tenure is None or tenure<date:
                    #no such user in user data
                    missinguser+=1
                    continue

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][0] < date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], 0, axis=0)
                userData[name]["regtasks"] = regtasks

                if len(regtasks[0]) == 0:
                    missinguser += 1
                    continue


                subtasks = userData[name]["subtasks"]
                while len(subtasks[0]) > 0 and subtasks[2][0] < date:
                    for l in range(len(subtasks)):
                        subtasks[l] = np.delete(subtasks[l], 0, axis=0)
                userData[name]["subtasks"] = subtasks

                if len(subtasks[0])==0:
                    missinguser+=1
                    continue

                #print("reg and sub history of",name,len(regtasks[0]),len(subtasks[0]))

                # reg history info
                regID, regDate = regtasks[0], regtasks[1]

                date_interval = regDate[len(regDate)-1] - date
                participate_recency = regDate[0]-date
                participate_frequency = len(regID)

                # sub history info
                subID, subNum, subDate, subScore, subrank = subtasks[0], subtasks[1], subtasks[2], subtasks[3], subtasks[4]

                commit_recency = subDate[0]-date
                commit_frequency = np.sum(subNum)
                last_perfromance = subScore[0]
                last_rank=subScore[0]
                win_indices = np.where(subrank == 0)[0]
                win_frequency = len(win_indices)
                if win_frequency==0:
                    #those without win history are filtered
                    missinguser+=1
                    continue
                win_recency = -1
                for i in range(len(subID)):
                    if subrank[i] == 0:
                        win_recency = subDate[i]
                        break

                user=[tenure-date,date_interval,participate_recency,participate_frequency,commit_recency,commit_frequency,
                      win_recency,win_frequency,last_perfromance,last_rank]
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

                #task vec
                task=task+list(self.selTasks.docX[taskPos])

                usernames.append(name)
                taskids.append(id)
                users.append(user)
                tasks.append(task)
                dates.append(date)

                if name in reg_usernams:
                    regists.append(1)
                else:
                    regists.append(0)

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
                    pickle.dump(data,f)
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
            pickle.dump(data,f)

        self.saveDataIndex(filepath=filepath,dataSegment=dataSegment+1)
        #print(self.tasktype+"=>:","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        #print()

def genDataSet():
    process_pools=[]
    with open("../data/TaskInstances/ClusterTaskIndex.data","rb") as f:
        tasktypes=pickle.load(f)

    with open("../data/runResults/types.txt","w") as f:
        for t in tasktypes:
            f.writelines(t+"\n")

        for t in tasktypes:

            proc=DataInstances(tasktype=t,cond=cond,usingmode=mode)
            proc.start()
            process_pools.append(proc)


if __name__ == '__main__':
    cond=multiprocessing.Condition()

    mode=0
    genDataSet()
