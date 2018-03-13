from DataPrepare.ConnectDB import *
import numpy as np
import pickle,json
import copy
import time
from DataPrepare.clusterTasks import onehotFeatures,showData
warnings.filterwarnings("ignore")
class Users:
    def __init__(self):
        self.loadData()

    def loadData(self):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select handle,memberage,skills,competitionNum,submissionNum,winNum from users'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.name=[]
        self.memberage=[]
        self.skills=[]
        self.competitionNum=[]
        self.submissionNum=[]
        self.winNum=[]
        for data in dataset:
            self.name.append(data[0])
            if data[1]<1:
                self.memberage.append(1)
            else:
                self.memberage.append(data[1])
            self.skills.append(data[2])
            self.competitionNum.append(data[3])
            self.submissionNum.append(data[4])
            self.winNum.append(data[5])
        self.name=np.array(self.name)
        self.skills=np.array(self.skills)
        self.memberage=np.log(self.memberage)
        self.competitionNum=np.array(self.competitionNum)
        self.submissionNum=np.array(self.submissionNum)
        self.winNum=np.array(self.winNum)

        print("users num=%d"%len(self.name))


    def getInfo(self,username):
        index=np.where(self.name==username)[0]
        if len(index)>0:
            return (self.memberage[index][0],self.skills[index][0])
        else:
            return (None,np.zeros(shape=len(self.skills[0])))

    def transformVec(self):
        n=len(self.name)
        X=np.concatenate((self.skills,np.reshape(self.memberage,newshape=(n,1))),axis=1)
        X=np.concatenate((X,np.reshape(self.winNum,newshape=(n,1))),axis=1)
        X=np.concatenate((X,np.reshape(self.competitionNum,newshape=(n,1))),axis=1)
        X=np.concatenate((X,np.reshape(self.submissionNum,newshape=(n,1))),axis=1)
        return X

class Registration:
    def __init__(self):
        self.loadData()
    def loadData(self):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid, handle,regdate from registration order by regdate desc'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.taskid=[]
        self.username=[]
        self.regdate=[]
        for data in dataset:
            self.taskid.append(data[0])
            self.username.append(data[1])
            if data[2]<1:
                self.regdate.append(1)
            else:
                self.regdate.append(data[2])
        self.taskid=np.array(self.taskid)
        self.username=np.array(self.username)
        self.regdate=np.log(np.array(self.regdate,dtype=np.int))
        print("registration num=%d"%len(dataset))
    def getTasks(self,username,curtime):
        indices=np.where(self.regdate>curtime)[0]
        if len(indices)==0:
            return None
        indices1=np.where(self.username[indices]==username)[0]
        if len(indices1)==0:
            return None
        indices=indices1+indices[0]
        ids=self.taskid[indices]
        date=self.regdate[indices]
        return (ids,date)

class Submission:
    def __init__(self):
        self.loadData()
    def loadData(self):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid,handle,subnum,submitdate,score from submission order by submitdate desc'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.taskid=[]
        self.username=[]
        self.subnum=[]
        self.subdate=[]
        self.score=[]

        for data in dataset:
            self.taskid.append(data[0])
            self.username.append(data[1])
            self.subnum.append(data[2])
            if data[3]>4200:
                self.subdate.append(4200)
            elif data[3]<1:
                self.subdate.append(1)
            else:
                self.subdate.append(data[3])
            if data[4] is not None:
                self.score.append(data[4])
            else:
                self.score.append(0)

        self.taskid=np.array(self.taskid)
        self.username=np.array(self.username)
        self.subnum=np.array(self.subnum)
        self.subdate=np.log(np.array(self.subdate,dtype=np.int))

        self.score=np.array(self.score)
        self.finalrank =np.zeros(shape=len(self.taskid))
        #assign ranking for each task
        taskset=set(self.taskid)
        for id in taskset:
            indices=np.where(self.taskid==id)[0]
            scores=copy.deepcopy(self.score[indices])
            count=0
            step=0
            #print(len(indices),scores)
            while count<10 and step<len(scores):
                step+=1
                max_score=np.max(scores)
                indices1=np.where(scores>=max_score)[0]
                #print(max_score, count,indices1)
                scores[indices1]=-1
                indices2=indices1+indices[0]
                self.finalrank[indices2]=count
                count += len(indices1)

        print("sub num=%d"%len(self.taskid))
    def getResultOfSubmit(self,username,taskid):
        indices=np.where(self.username==username)[0]
        if len(indices)==0:
            return None
        indices1=np.where(self.taskid[indices]==taskid)[0]
        if len(indices1)==0:
            return None
        indices=indices1+indices[0]
        return (self.subnum[indices][0],self.finalrank[indices][0])
    def getTasks(self,username,curtime):
        indices=np.where(self.subdate>curtime)[0]
        if len(indices)==0:
            return None
        indices1=np.where(self.username[indices]==username)[0]
        if len(indices1)==0:
            return None
        indices=indices1+indices[0]
        ids=self.taskid[indices]
        subnum=self.subnum[indices]
        date=self.subdate[indices]
        score=self.score[indices]
        rank=self.finalrank[indices]
        return (ids,subnum,date,score,rank)

class GlobalSubmitTestInstances:
    def __init__(self,regdata,subdata,userdata):
        self.regdata=regdata
        self.subdata=subdata
        self.userdata=userdata
    def createVec(self,choice=1):
        tasks=[]
        users=[]
        with open("../data/clusterResult/taskVec"+str(choice)+".data","rb") as f:
            taskdata=json.load(f)
        missingtask=0
        missinguser=0
        usernames=[]
        taskids=[]
        dates=[]
        submits=[]
        ranks=[]
        t0=time.time()
        for _ in range(len(regs.taskid)):
            if (_+1)%10000==0:
                print(_+1,"of",len(regs.taskid),"in %ds"%(time.time()-t0))
                t0=time.time()
            id, name, date=self.regdata.taskid[_],self.regdata.username[_],self.regdata.regdate[_]
            #print("task-user", id, name, date)
            if str(id) not in taskdata.keys():
                #print(id,"not in data set")
                missingtask+=1
                continue
            task = taskdata[str(id)]
            regtasks=self.regdata.getTasks(name,date)
            #print("reg",regtasks)
            if regtasks is None:
                missinguser+=1
                continue
            regID,regDate=regtasks[0],regtasks[1]
            participate_recency=regDate[len(regDate)-1]
            date_interval=max(1,regDate[0]-regDate[len(regDate)-1])
            participate_frequency=len(regID)/date_interval

            subtasks=self.subdata.getTasks(name,date)
            #print("sub",subtasks)
            if subtasks is None:
                commit_recency=regDate[0]*2
                commit_frequency=0
                last_perfromance=0
                win_recency=commit_recency*2
                win_frequency=0

            else:
                subID,subNum,subDate,subScore,subrank=subtasks[0],subtasks[1],subtasks[2],subtasks[3],subtasks[4]
                commit_recency=subDate[len(subDate)-1]
                commit_frequency=np.sum(subNum)/date_interval
                last_perfromance=subScore[len(subScore)-1]
                win_recency=commit_recency*2
                for i in range(1,len(subID)+1):
                    if subrank[-i]==0:
                        win_recency=subDate[-i]
                win_indices=np.where(subrank==0)[0]
                if len(win_indices)>0:
                    win_frequency=len(win_indices)/date_interval
                else:
                    win_frequency=0

            tenure,skills=self.userdata.getInfo(name)
            #print("user",tenure,skills)
            if tenure is None:
                tenure=regDate[0]
            user=[tenure,participate_recency,participate_frequency,commit_recency,commit_frequency,
                  win_recency,win_frequency,last_perfromance]+skills.tolist()

            usernames.append(name)
            taskids.append(id)
            users.append(user)
            tasks.append(task)
            dates.append(date)
            curPerformance=subs.getResultOfSubmit(name,id)
            #print(curPerformance)
            if curPerformance is not None:
                submits.append(curPerformance[0])
                ranks.append(curPerformance[1])
            else:
                submits.append(0)
                ranks.append(10)


        print("missing task",missingtask,"missing user",missinguser,"instances size",len(taskids))
        print()
        data={}
        for i in range(len(taskids)):
            data["usernames"]=usernames
            data["taskids"]=taskids
            data["tasks"]=tasks
            data["users"]=users
            data["dates"]=dates
            data["submits"]=submits
            data["ranks"]=ranks

        print("saving data")
        with open("../data/Instances/task_user"+str(choice)+".data","wb") as f:
            pickle.dump(data,f)

        return data


if __name__ == '__main__':
    user=Users()
    user.skills,features=onehotFeatures(user.skills)
    print("encoding skills feature_num=",features)
    regs=Registration()
    subs=Submission()
    '''
    x=np.array(user.memberage).reshape((len(user.memberage),1)).tolist()
    showData(x)
    showData(np.log(x).tolist())
    x = np.array(regs.regdate,dtype=np.int).reshape((len(regs.regdate), 1)).tolist()
    showData(x)
    showData(np.log(x).tolist())
    x = np.array(subs.subdate,dtype=np.int).reshape((len(subs.subdate), 1)).tolist()
    showData(x)
    showData(np.log(x).tolist())
    x = np.array(subs.score, dtype=np.int).reshape((len(subs.score), 1)).tolist()
    showData(x)
    showData(np.log(x).tolist())
    '''


    gInst=GlobalSubmitTestInstances(regs,subs,user)
    data=gInst.createVec(1)
    X=np.concatenate((data["tasks"],data["users"]),axis=1)
    [print(x) for x in X[:3]]

