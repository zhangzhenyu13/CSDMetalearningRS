import numpy as np
import pickle
import time,gc
from DataPrepare.DataContainer import *

class DataInstances:
    def __init__(self,tasktype,choice):
        self.tasktype=tasktype
        self.choice=choice
        self.loadData(choice)

    def loadData(self,choice):
        #load task data
        with open("../data/TaskInstances/taskDataSet/"+self.tasktype+"-taskData-"+str(choice)+".data","rb") as f:
            data=pickle.load(f)
            tasks=data["tasks"]
            taskids=data["taskids"]
            self.taskdata={}
            for i in range(len(taskids)):
                self.taskdata[taskids[i]]=tasks[i]
        #load user data
        self.userdata=UserHistoryGenerator()

        #load reg data
        with open("../data/TaskInstances/RegInfo/"+self.tasktype+"-regs-"+str(choice)+".data","rb") as f:
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
        with open("../data/TaskInstances/SubInfo/"+self.tasktype+"-subs-"+str(choice)+".data","rb") as f:
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

    def createInstancesWithRegHistoryInfo(self,filepath=None,threshold=1e+6,verboseNum=100,runPID=None):
        if filepath is None:
            filepath="../data/TopcoderDataSet/regHistoryBasedData/"+self.tasktype+"-user_task-"+str(self.choice)+".data"

        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]
        taskdata=self.taskdata

        userData=self.userdata.loadActiveUserHistory(tasktype=self.tasktype,mode=0)

        print(self.tasktype+"=>runPID",runPID,":","construct registration history based instances with %d tasks and %d users" %
              (len(taskdata.keys()), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.regdata.taskids)):
            if (index+1)%verboseNum==0:
                print("runPID",runPID,":",index+1,"of",len(self.regdata.taskids),
                      "current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                t0=time.time()

            id,date=self.regdata.taskids[index],self.regdata.regdates[index]

            # task data of id
            task = taskdata[id]

            reg_usernams, regDates=self.regdata.getRegUsers(id)
            if reg_usernams is None:
                missingtask+=1
                continue

            for name in userData.keys():

                tenure, skills = userData[name]["tenure"],userData[name]["skills"]

                if tenure is None:
                    #no such user in user data
                    missinguser+=1
                    continue

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][len(regtasks[1]) - 1] < date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], len(regtasks[l]) - 1, axis=0)
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

                date_interval = regDate[0] - date
                participate_recency = regDate[len(regDate) - 1]-date
                participate_frequency = len(regID)


                user=[tenure-date,date_interval,participate_recency,participate_frequency]+skills.tolist()

                usernames.append(name)
                taskids.append(id)
                users.append(user)
                tasks.append(task)
                dates.append(date)

            if len(taskids)>threshold:
                data={}
                print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
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

        print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
        with open(filepath+str(dataSegment),"wb") as f:
            pickle.dump(data,f)

        print("runPID",runPID,":","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()

        return data

    def createInstancesWithSubHistoryInfo(self,filepath=None,threshold=1e+6,verboseNum=100,runPID=None):
        if filepath is None:
            filepath="../data/TopcoderDataSet/subHistoryBasedData/"+self.tasktype+"-user_task-"+str(self.choice)+".data"

        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]

        taskdata=self.taskdata

        userData=self.userdata.loadActiveUserHistory(mode=1,tasktype=self.tasktype)

        print(self.tasktype+"=>runPID",runPID,":","construct submission history based instances with %d tasks and %d users" %
              (len(taskdata), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.regdata.taskids)):
            if (index+1)%verboseNum==0:
                print("runPID",runPID,":",index+1,"of",len(self.regdata.taskids),
                      "current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                print()
                t0=time.time()

            id,date=self.regdata.taskids[index],self.regdata.regdates[index]

            # task data of id
            task = taskdata[id]

            reg_usernams, regDates=self.regdata.getRegUsers(id)

            if reg_usernams is None:
                missingtask+=1
                continue
            for name in userData.keys():

                tenure, skills = userData[name]["tenure"],userData[name]["skills"]

                if tenure is None:
                    #no such user in user data
                    missinguser+=1
                    continue

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][len(regtasks[1]) - 1] < date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], len(regtasks[l]) - 1, axis=0)
                userData[name]["regtasks"] = regtasks

                if len(regtasks[0]) == 0:
                    missinguser += 1
                    continue


                subtasks = userData[name]["subtasks"]
                while len(subtasks[0]) > 0 and subtasks[2][len(subtasks[2]) - 1] < date:
                    for l in range(len(subtasks)):
                        subtasks[l] = np.delete(subtasks[l], len(subtasks[l]) - 1, axis=0)
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

                date_interval = regDate[0] - date
                participate_recency = regDate[len(regDate) - 1]-date
                participate_frequency = len(regID)

                # sub history info
                subID, subNum, subDate, subScore, subrank = subtasks[0], subtasks[1], subtasks[2], subtasks[3], subtasks[4]

                commit_recency = subDate[len(subDate) - 1]-date
                commit_frequency = np.sum(subNum)
                last_perfromance = subScore[len(subScore) - 1]
                last_rank=subScore[len(subrank)-1]

                user=[tenure-date,date_interval,participate_recency,participate_frequency,commit_recency,commit_frequency,
                      last_perfromance,last_rank]+skills.tolist()

                usernames.append(name)
                taskids.append(id)
                users.append(user)
                tasks.append(task)
                dates.append(date)

            if len(taskids)>threshold:
                data={}
                print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
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



        print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
        with open(filepath+str(dataSegment),"wb") as f:
            pickle.dump(data,f)

        print("runPID",runPID,":","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()
        return data

    def createInstancesWithWinHistoryInfo(self,filepath=None,threshold=1e+6,verboseNum=100,runPID=None):
        if filepath is None:
            filepath="../data/TopcoderDataSet/winHistoryBasedData/"+self.tasktype+"-user_task-"+str(self.choice)+".data"

        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]

        taskdata=self.taskdata

        userData=self.userdata.loadActiveUserHistory(tasktype=self.tasktype,mode=2)

        print(self.tasktype+"=>runPID",runPID,":","construct winning history based instances with %d tasks and %d users" %
              (len(taskdata), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.regdata.taskids)):
            if (index+1)%verboseNum==0:
                print("runPID",runPID,":",index+1,"of",len(self.regdata.taskids),
                      "current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                t0=time.time()

            id,date=self.regdata.taskids[index],self.regdata.regdates[index]

            # task data of id
            task = taskdata[id]

            reg_usernams, regDates=self.regdata.getRegUsers(id)

            if reg_usernams is None:
                missingtask+=1
                continue
            for name in userData.keys():

                tenure, skills = userData[name]["tenure"],userData[name]["skills"]

                if tenure is None:
                    #no such user in user data
                    missinguser+=1
                    continue

                #get reg and sub history before date for user:name
                regtasks = userData[name]["regtasks"]
                while len(regtasks[0]) > 0 and regtasks[1][len(regtasks[1]) - 1] < date:
                    for l in range(len(regtasks)):
                        regtasks[l] = np.delete(regtasks[l], len(regtasks[l]) - 1, axis=0)
                userData[name]["regtasks"] = regtasks

                if len(regtasks[0]) == 0:
                    missinguser += 1
                    continue


                subtasks = userData[name]["subtasks"]
                while len(subtasks[0]) > 0 and subtasks[2][len(subtasks[2]) - 1] < date:
                    for l in range(len(subtasks)):
                        subtasks[l] = np.delete(subtasks[l], len(subtasks[l]) - 1, axis=0)
                userData[name]["subtasks"] = subtasks

                if len(subtasks[0])==0:
                    missinguser+=1
                    continue

                #print("reg and sub history of",name,len(regtasks[0]),len(subtasks[0]))

                # reg history info
                regID, regDate = regtasks[0], regtasks[1]

                date_interval = regDate[0] - date
                participate_recency = regDate[len(regDate) - 1]-date
                participate_frequency = len(regID)

                # sub history info
                subID, subNum, subDate, subScore, subrank = subtasks[0], subtasks[1], subtasks[2], subtasks[3], subtasks[4]

                commit_recency = subDate[len(subDate) - 1]-date
                commit_frequency = np.sum(subNum)
                last_perfromance = subScore[len(subScore) - 1]
                last_rank=subScore[len(subrank)-1]
                win_indices = np.where(subrank == 0)[0]
                win_frequency = len(win_indices)
                if win_frequency==0:
                    #those without win history are filtered
                    missinguser+=1
                    continue
                win_recency = -1
                for i in range(1, len(subID) + 1):
                    if subrank[-i] == 0:
                        win_recency = subDate[-i]
                        break

                user=[tenure,date_interval,participate_recency,participate_frequency,commit_recency,commit_frequency,
                      win_recency,win_frequency,last_perfromance,last_rank]+skills.tolist()

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

            if len(taskids)>threshold:
                data={}
                print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
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

        print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
        with open(filepath+str(dataSegment),"wb") as f:
            pickle.dump(data,f)

        print("runPID",runPID,":","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()



if __name__ == '__main__':
    choice=1
    with open("../data/TaskInstances/OriginalTasktype.data","rb") as f:
        tasktypes=pickle.load(f)
        for t in tasktypes.keys():
            if len(tasktypes[t])<50:
                continue

            taskids=tasktypes[t]
            tasktype=t.replace("/","_")
            DataInstances(tasktype=tasktype,choice=choice).createInstancesWithRegHistoryInfo()
    pass
