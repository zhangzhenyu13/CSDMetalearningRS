from DataPrepare.ConnectDB import *
import numpy as np
import pickle
import copy
import time
import gc,multiprocessing
from ML_Models.Model_def import FilePath
from Utility.personalizedSort import  MySort
from Utility.FeatureEncoder import onehotFeatures
warnings.filterwarnings("ignore")
filep=FilePath()



class DataInstances:
    def __init__(self,regdata,subdata,userdata,taskIndex):
        self.regdata=regdata
        self.subdata=subdata
        self.userdata=userdata
        self.taskIndex=taskIndex

    def setLocality(self,taskIDs=None):

        self.activeReg=[]
        taskIndex=self.taskIndex

        if taskIDs is None:
            for i in range(len(taskIndex.taskIDs)):
                self.activeReg.append((taskIndex.taskIDs[i],taskIndex.postingdate[i]))
        else:
            for id in taskIDs:
                indices=np.where(taskIndex.taskIDs==id)[0]
                if len(indices)>0:
                    self.activeReg.append((id,taskIndex.postingdate[indices][0]))

        print("set locality: regtasks size=%d" % (len(self.activeReg)))

    def createInstancesWithRegHistoryInfo(self,filepath=None,threshold=1e+6,verboseNum=1000,runPID=None):
        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]
        taskdata=self.taskIndex.taskdata

        userData=self.userdata.loadActiveUserHistory(mode=0)

        print("runPID",runPID,":","construct registration history based instances with %d tasks and %d users" %
              (len(taskdata), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.activeReg)):
            if (index+1)%verboseNum==0:
                print("runPID",runPID,":",index+1,"of",len(self.activeReg),"current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                t0=time.time()

            id,date=self.activeReg[index]

            # task data of id
            task = taskdata[id]

            reg_usernams, regDates=self.regdata.getUsers(id)
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

            if filepath is not None and len(taskids)>threshold:
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
                    data["tasks"] = addTaskInfo(taskids=data["taskids"],X=data["tasks"])
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

        if filepath is not None:
            print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
            with open(filepath+str(dataSegment),"wb") as f:
                data["tasks"] = addTaskInfo(taskids=data["taskids"],X=data["tasks"])
                pickle.dump(data,f)

        print("runPID",runPID,":","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()

        return data

    def createInstancesWithSubHistoryInfo(self,filepath=None,threshold=1e+6,verboseNum=1000,runPID=None):
        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]

        taskdata=self.taskIndex.taskdata

        userData=self.userdata.loadActiveUserHistory(mode=1)

        print("runPID",runPID,":","construct submission history based instances with %d tasks and %d users" %
              (len(taskdata), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.activeReg)):
            if (index+1)%verboseNum==0:
                print("runPID",runPID,":",index+1,"of",len(self.activeReg),"current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                print()
                t0=time.time()

            id,date=self.activeReg[index]

            # task data of id
            task = taskdata[id]

            reg_usernams, regDates=self.regdata.getUsers(id)

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

            if filepath is not None and len(taskids)>threshold:
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
                    data["tasks"] = addTaskInfo(taskids=data["taskids"],X=data["tasks"])
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


        if filepath is not None:
            print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
            with open(filepath+str(dataSegment),"wb") as f:
                data["tasks"] = addTaskInfo(taskids=data["taskids"],X=data["tasks"])
                pickle.dump(data,f)

        print("runPID",runPID,":","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()
        return data

    def createInstancesWithWinHistoryInfo(self,filepath=None,threshold=1e+6,verboseNum=1000,runPID=None):
        tasks=[]
        users=[]
        usernames = []
        taskids = []
        dates = []
        submits = []
        ranks = []
        scores=[]
        regists=[]

        taskdata=self.taskIndex.taskdata

        userData=self.userdata.loadActiveUserHistory(mode=2)

        print("runPID",runPID,":","construct winning history based instances with %d tasks and %d users" %
              (len(taskdata), len(userData.keys())))

        missingtask=0
        missinguser=0
        dataSegment=0
        t0=time.time()

        for index in range(len(self.activeReg)):
            if (index+1)%verboseNum==0:
                print("runPID",runPID,":",index+1,"of",len(self.activeReg),"current size=%d"%(len(taskids)),"in %ds"%(time.time()-t0))
                print("registered =%d/%d"%(np.sum(regists),len(regists)))
                t0=time.time()

            id,date=self.activeReg[index]

            # task data of id
            task = taskdata[id]

            reg_usernams, regDates=self.regdata.getUsers(id)

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

            if filepath is not None and len(taskids)>threshold:
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
                    data["tasks"] = addTaskInfo(taskids=data["taskids"],X=data["tasks"])
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

        if filepath is not  None:
            print("runPID",runPID,":","saving %d Instances, segment=%d"%(len(taskids),dataSegment))
            with open(filepath+str(dataSegment),"wb") as f:
                data["tasks"] = addTaskInfo(taskids=data["taskids"],X=data["tasks"])
                pickle.dump(data,f)

        print("runPID",runPID,":","missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()

        return data

#add tasktype info to global
def addTaskInfo(taskids,X):
    with open(filep.getClusterFilePath(local=2,choice=1), "rb") as f:
        dataSet = pickle.load(f)
    taskType={}
    for key in dataSet.keys():
        for id in dataSet[key]:
            taskType[id]=key
    typeInfo=[]
    for id in taskids:
        typeInfo.append(taskType[id])
    typeInfo,_=onehotFeatures(typeInfo)
    X=np.concatenate((typeInfo,X),axis=1)
    return X

def genRegisteredInstances(gInst,local,mode,runPID):

    choice = gInst.taskIndex.choice

    print("Process Parameters:","choice=", choice, "; local status=", local,"; mode=",mode)
    genMethod={
        0:gInst.createInstancesWithRegHistoryInfo,
        1:gInst.createInstancesWithSubHistoryInfo,
        2:gInst.createInstancesWithWinHistoryInfo
    }

    if local>0:

        with open(filep.getClusterFilePath(local=local,choice=choice), "rb") as f:
            taskidClusters = pickle.load(f)

        dataClusters = {}
        print("creating train Local data")
        for k in taskidClusters.keys():
            print("creating instances for cluster(%d):" % (len(taskidClusters[k])), k)
            gInst.setLocality(taskidClusters[k])
            data = genMethod[mode]()
            dataClusters[k] = data
            print()

        with open(filep.getInstancesFilePath(local=local,mode=mode,choice=choice), "wb") as f:
            pickle.dump(dataClusters, f)

    else:
        print("creating global data")
        gInst.setLocality()
        genMethod[mode](filepath=filep.getInstancesFilePath(local=local,mode=mode,choice=choice),verboseNum=100,runPID=runPID)
        print()


if __name__ == '__main__':
    choice=1
    tasks=Tasks(choice=choice)
    #print(tasks.taskIDs[:10],tasks.postingdate[:10])
    regs = Registration(tasks.taskIDs)
    subs = Submission(tasks.taskIDs)

    user = Users()
    user.skills, features = onehotFeatures(user.skills,threshold_num=100)

    print("encoding skills feature_num=", features)
    users=ActiveUserHistory(userdata=user,regdata=regs,subdata=subs)
    #data=users.loadActiveUserHistory(mode=2)
    #for name in data.keys():
    #    print(name,data[name]["regtasks"][:10],data[name]["subtasks"],data[name]["tenure"])
    #exit(10)

    for mode in ():
        multiprocessing.Process(target=users.genActiveUserHistory,args=(mode,)).start()
        #users.genActiveUserHistory(mode=mode)
    #exit(10)

    gInst = DataInstances(userdata=users,regdata=regs,subdata=subs,taskIndex=tasks)
    pid=1
    for local,mode in ((0,0),(0,1),(0,2)):
        multiprocessing.Process(target=genRegisteredInstances,args=(gInst,local,mode,pid)).start()
        pid+=1
        #genRegisteredInstances(gInst,local=local,mode=mode)

