from ML_Models.DocTopicsModel import LSAFlow,LDAFlow
from DataPrepare.ConnectDB import ConnectDB
from Utility.FeatureEncoder import onehotFeatures
import numpy as np
import pickle

class TaskDataContainer:
    def __init__(self,typename):
        self.ids=[]
        self.docs=[]
        self.techs=[]
        self.lans=[]
        self.startdates=[]
        self.durations=[]
        self.prizes=[]
        self.diffdegs=[]
        self.taskType=typename
    def encodingFeature(self,choice):
        self.choice=choice

        docsEncoder={1:LSAFlow,2:LDAFlow}
        print("encoding docs",self.taskType)
        doc_model=docsEncoder[choice]()
        try:
            doc_model.loadModel()
        except FileNotFoundError as e:
            print(e,"now train model")
            doc_model.train_doctopics(self.docs)

        self.docs=doc_model.transformVec(self.docs)
        print(self.taskType,"docs shape",self.docs.shape)

        print("encoding techs",self.taskType)
        self.techs=onehotFeatures(self.techs)
        print(self.taskType,"docs shape",self.techs.shape)

        print("encoding lans",self.taskType)
        self.lans=onehotFeatures(self.lans)
        print(self.taskType,"docs shape",self.lans.shape)

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
        self.memberage=np.array(self.memberage)
        self.competitionNum=np.array(self.competitionNum)
        self.submissionNum=np.array(self.submissionNum)
        self.winNum=np.array(self.winNum)

        print("users num=%d"%len(self.name))

    def getUsers(self):
        names=self.name
        return names

    def getInfo(self,username):
        index=np.where(self.name==username)[0]
        if len(index)>0:
            #print(index,self.memberage[index][0],self.skills[index][0])
            return (self.memberage[index][0],self.skills[index][0])
        else:
            return (None,None)


class Registration:
    def __init__(self,taskIDs):
        self.loadData(taskIDs)

    def loadData(self,taskIDs):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid, handle,regdate from registration order by regdate desc'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.taskid=[]
        self.username=[]
        self.regdate=[]
        for data in dataset:
            if data[0] not in taskIDs:
                continue
            self.taskid.append(data[0])
            self.username.append(data[1])
            if data[2]<1:
                self.regdate.append(1)
            else:
                self.regdate.append(data[2])
        self.taskid=np.array(self.taskid)
        self.username=np.array(self.username)

        self.regdate=np.array(self.regdate,dtype=np.int)
        print("registration num=%d"%len(dataset),"user num=%d"%len(set(self.username)),"task num=%d"%len(set(self.taskid)))

    def getUserHistory(self,username):

        indices=np.where(self.username==username)[0]
        if len(indices)==0:
            return (np.array([]),np.array([]))

        ids=self.taskid[indices]
        date=self.regdate[indices]
        return [ids,date]

    def getUsers(self,taskid):
        indices=np.where(self.taskid==taskid)[0]
        if len(indices)==0:
            return None,None
        #print(indices)
        #print(len(self.username))
        taskUsers=self.username[indices]
        regDates=self.regdate[indices]
        return taskUsers,regDates
    def setActiveTaskUser(self,taskids=None,usernames=None):

        if taskids is not None:
            activeID = []
            activeName = []
            activeDate = []
            for id in taskids:
                indices=np.where(self.taskid==id)[0]
                if len(indices)==0:
                    continue
                else:
                    ids=self.taskid[indices].tolist()
                    dates=self.regdate[indices].tolist()
                    names=self.username[indices].tolist()
                    activeID=activeID+ids
                    activeName=activeName+names
                    activeDate=activeDate+dates

            self.taskid=np.array(activeID)
            self.username=np.array(activeName)
            self.regdate=np.array(activeDate)

        if usernames is not None:
            activeID = []
            activeName = []
            activeDate = []
            for name in usernames:
                indices=np.where(self.username==name)[0]
                if len(indices)==0:
                    continue
                else:
                    ids = self.taskid[indices].tolist()
                    dates = self.regdate[indices].tolist()
                    names = self.username[indices].tolist()
                    activeID = activeID + ids
                    activeName = activeName + names
                    activeDate = activeDate + dates
            self.taskid = np.array(activeID)
            self.username = np.array(activeName)
            self.regdate = np.array(activeDate)
        #data active
        print("active reg data set size=",len(self.taskid))

class Submission:
    def __init__(self,taskIDs):
        self.loadData(taskIDs)
    def loadData(self,taskIDs):
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
            if data[0] not in taskIDs:
                continue
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
        self.subdate=np.array(self.subdate,dtype=np.int)

        self.score=np.array(self.score)
        self.finalrank =np.zeros(shape=len(self.taskid))
        #assign ranking for each task
        #count=0

        taskset=set(self.taskid)
        #print("init %d taskid"%len(taskset))
        for id in taskset:
            #if (count+1)%1000==0:
            #    print("init rank %d/%d"%(count+1,len(taskset)))
            #    count+=1
            indices=np.where(self.taskid==id)[0]
            scores=copy.deepcopy(self.score[indices])
            X=[]
            for i in range(len(scores)):
                X.append((indices[i],scores[i]))
                #print(X[i])
            m_s=MySort(X)
            m_s.compare_vec_index=-1
            X=m_s.mergeSort()
            #print("after sort")
            #for i in range(len(X)):
            #    print(X[i])
            #print()
            p=0
            rank=0
            while rank<10 and p<len(X):
                if X[p][1]==0:
                    break
                for j in range(p,len(X)):
                    if X[j][1]!=X[p][1]:
                        for i in range(p,j):
                            self.finalrank[X[p][0]]=rank
                        rank+=j-p
                        p=j
                        break
                    if j==len(X)-1:
                        for i in range(p,len(X)):
                            self.finalrank[X[p][0]]=rank
                        rank+=len(X)-p
                        p=len(X)
                        break
                if p==len(X)-1:
                    if rank>10:
                        rank=10
                    self.finalrank[X[p][0]]=rank
                    p+=1
                    break

            if p<len(X):
                for j in range(p,len(X)):
                    self.finalrank[X[j][0]]=10


        print("sub num=%d"%len(self.taskid),"user num=%d"%len(set(self.username)),"task num=%d"%len(set(self.taskid)))

    def getResultOfSubmit(self,username,taskid):
        indices=np.where(self.username==username)[0]
        if len(indices)==0:
            return None
        indices1=np.where(self.taskid[indices]==taskid)[0]
        if len(indices1)==0:
            return None
        indices=indices1+indices[0]
        return [self.subnum[indices][0],self.finalrank[indices][0],self.score[indices][0]]
    def getUsers(self,taskid):
        indices = np.where(self.taskid == taskid)[0]
        if len(indices) == 0:
            return None,None
        # print(indices)
        # print(len(self.username))
        taskUsers=self.username[indices]
        taskDates=self.subdate[indices]
        return taskUsers,taskDates

    def getUserHistory(self,username):

        indices=np.where(self.username==username)[0]
        if len(indices)==0:
            return (np.array([]),np.array([]),np.array([]),np.array([]),np.array([]))

        ids=self.taskid[indices]
        subnum=self.subnum[indices]
        date=self.subdate[indices]
        score=self.score[indices]
        rank=self.finalrank[indices]
        return (ids,subnum,date,score,rank)

    def setActiveTaskUser(self,taskids=None, usernames=None):
        #set submission entry related with given taskids and usernames
        if taskids is not None:
            activeID = []
            activeName = []
            activeDate = []
            activeNum=[]
            activeScore=[]
            activeRank=[]

            for id in taskids:
                indices = np.where(self.taskid == id)[0]
                if len(indices) == 0:
                    continue
                else:
                    ids = self.taskid[indices].tolist()
                    dates = self.subdate[indices].tolist()
                    names = self.username[indices].tolist()
                    nums=self.subnum[indices].tolist()
                    scores=self.score[indices].tolist()
                    ranks=self.finalrank[indices].tolist()

                    activeID = activeID + ids
                    activeName = activeName + names
                    activeDate = activeDate + dates
                    activeNum=activeNum+nums
                    activeScore=activeScore+scores
                    activeRank=activeRank+ranks

            self.taskid = np.array(activeID)
            self.username = np.array(activeName)
            self.subdate = np.array(activeDate)
            self.subnum=np.array(activeNum)
            self.score=np.array(activeScore)
            self.finalrank=np.array(activeRank)

        if usernames is not None:
            activeID = []
            activeName = []
            activeDate = []
            activeNum = []
            activeScore = []
            activeRank = []
            for name in usernames:
                indices = np.where(self.username == name)[0]
                if len(indices) == 0:
                    continue
                else:
                    #print(self.taskid)
                    #print(indices)
                    ids = self.taskid[indices].tolist()
                    dates = self.subdate[indices].tolist()
                    names = self.username[indices].tolist()
                    nums = self.subnum[indices].tolist()
                    scores = self.score[indices].tolist()
                    ranks = self.finalrank[indices].tolist()

                    activeID = activeID + ids
                    activeName = activeName + names
                    activeDate = activeDate + dates
                    activeNum = activeNum + nums
                    activeScore = activeScore + scores
                    activeRank = activeRank + ranks

            self.taskid = np.array(activeID)
            self.username = np.array(activeName)
            self.subdate = np.array(activeDate)
            self.subnum = np.array(activeNum)
            self.score = np.array(activeScore)
            self.finalrank = np.array(activeRank)

        #sort data date desc

        # data active
        print("active sub data set size=", len(self.taskid))

class Tasks:
    def __init__(self,choice=1,begindate=5000):
        with open("../data/clusterResult/taskVec" + str(choice) + ".data", "rb") as f:
            taskdata = pickle.load(f)
            ids = taskdata["taskids"]
            X = taskdata["tasks"]
            print("task vec data size=%d"%(len(ids)))
            taskdata = {}
            for i in range(len(ids)):
                taskdata[ids[i]] = X[i]

        self.taskdata=taskdata
        self.choice=choice
        self.taskIDs=[]
        self.postingdate=[]
        self.loadData(begindate)
    def loadData(self,begindate=5000):

        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd="select taskid, postingdate from task where postingdate <="+str(begindate)+" and postingdate>=0 order by postingDate asc;"
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        for data in dataset:
            if data[0] not in self.taskdata.keys():
                continue
            self.taskIDs.append(data[0])
            self.postingdate.append(data[1])

        self.taskIDs=np.array(self.taskIDs)
        self.postingdate=np.array(self.postingdate,dtype=np.int)
        print("task item size=",len(self.taskIDs))

class ActiveUserHistory:
    def __init__(self,userdata,regdata,subdata):
        self.userdata=userdata
        self.regdata=regdata
        self.subdata=subdata
        self.tag={0:"Reg",1:"Sub",2:"Win"}

    def genActiveUserHistory(self,mode=0):
        user,regdata,subdata=self.userdata,self.regdata,self.subdata

        usernames=user.getUsers()
        userData = {}
        for username in usernames:
            regids, regdates = regdata.getUserHistory(username)
            if len(regids) == 0:
                #default for those have registered
                continue
            subids, subnum, subdates, score, rank = subdata.getUserHistory(username)
            winindices=np.where(rank==0)[0]
            if mode==1 and np.sum(subnum)<1:
                #for those ever submitted
                continue
            if mode==2 and len(winindices)==0:
                #for those ever won
                continue

            tenure,skills=user.getInfo(username)
            userData[username] = {"regtasks": [regids, regdates],
                                  "subtasks": [subids, subnum, subdates, score, rank],
                                  "tenure":tenure,"skills":skills}
            #print(username, "sub histroy and reg histrory=", len(userData[username]["subtasks"][0]),
            #      len(userData[username]["regtasks"][0]))

        print("saving history of %d active users" % len(userData))
        with open("../data/Instances/UserHistory/activeUsers"+self.tag[mode]+".data", "wb") as f:
            pickle.dump(userData, f)

    def loadActiveUserHistory(self,mode=0):
        print("loading history of active users")
        with open("../data/Instances/UserHistory/activeUsers"+self.tag[mode]+".data", "rb") as f:
            act_userData = pickle.load(f)

        return act_userData
