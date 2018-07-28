from ML_Models.DocTopicsModel import LDAFlow,LSAFlow
from DataPrepare.ConnectDB import ConnectDB
from Utility.FeatureEncoder import *
from Utility.TagsDef import  *
import _pickle as pickle,copy
import warnings
warnings.filterwarnings("ignore")

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


        print("loading encoding data",self.taskType)
        with open("../data/TaskInstances/GlobalEncoding.data","rb") as f:
            encoder=pickle.load(f)
        techs_dict=encoder["techs"]
        lans_dict=encoder["lans"]

        docsEncoder={1:LDAFlow,2:LSAFlow}
        doc_model=docsEncoder[choice]()
        doc_model.name="global"
        try:
            doc_model.loadModel()
        except :
            print("loading doc model failed, now begin to train the model")
            doc_model.name=self.taskType
            doc_model.train_doctopics(self.docs)
        finally:
            print("model loaded")
            print()
        self.docs=doc_model.transformVec(self.docs)
        print(self.taskType,"docs shape",self.docs.shape)

        print("encoding techs",self.taskType)
        self.techs_vec=EncodeByDict(self.techs,techs_dict,TaskTechs)
        print(self.taskType,"techs shape",self.techs_vec.shape)

        print("encoding lans",self.taskType)
        self.lans_vec=EncodeByDict(self.lans,lans_dict,TaskLans)
        print(self.taskType,"lans shape",self.lans_vec.shape)

class UserDataContainer:

    def __init__(self,names,tenure,skills):
        self.names=np.array(names)
        self.tenure=np.array(tenure)
        self.skills=np.array(skills)
        self.skills_vec=None
        print("User Instances size=%d"%len(self.names))

    def getUserInfo(self,username):
        index=np.where(self.names==username)[0]
        if len(index)>0:
            #print(index,self.memberage[index][0],self.skills[index][0])
            return (self.tenure[index][0],self.skills[index][0],self.skills_vec[index][0])
        else:
            return (None,None,None)

class RegistrationDataContainer:
    def __init__(self,tasktype,taskids,usernames,regdates):
        self.taskids=np.array(taskids)
        self.names=np.array(usernames)
        self.regdates=np.array(regdates)
        self.tasktype=tasktype

        print("reg data of",tasktype+",size=%d"%len(self.taskids))

    def getRegUsers(self,taskid):
        indices=np.where(self.taskids==taskid)[0]
        if len(indices)==0:
            return None,None
        #print(indices)
        #print(len(self.username))
        regUsers=self.names[indices]
        regDates=self.regdates[indices]
        return regUsers,regDates

    def getAllUsers(self):
        usernames=set(self.names)
        return usernames

    def getUserHistory(self,username):
        indices=np.where(self.names==username)[0]
        if len(indices)==0:
            return (np.array([]),np.array([]))

        ids=self.taskids[indices]
        dates=self.regdates[indices]
        return [ids,dates]

class SubmissionDataContainer:
    def __init__(self,tasktype,taskids,usernames,subnums,subdates,scores,finalranks):
        self.taskids=np.array(taskids)
        self.names=np.array(usernames)
        self.subnums=np.array(subnums)
        self.subdates=np.array(subdates)
        self.scores=np.array(scores)
        self.finalranks=np.array(finalranks)

        print("submission data of "+tasktype+", size=%d"%len(self.taskids))

    def getSubUsers(self,taskid):
        indices = np.where(self.taskids == taskid)[0]
        if len(indices) == 0:
            return None,None
        # print(indices)
        # print(len(self.username))
        subUsers=self.names[indices]
        subDates=self.subdates[indices]
        return subUsers,subDates

    def getAllUsers(self):
        usernames=set(self.names)
        return usernames

    def getResultOfSubmit(self,username,taskid):
        indices=np.where(self.names==username)[0]
        #print(indices);exit(10)
        if len(indices)==0:
            return None
        indices1=np.where(self.taskids[indices]==taskid)[0]
        if len(indices1)==0:
            return None

        index=indices[indices1[0]]

        return [self.subnums[index],self.finalranks[index],self.scores[index]]

    def getUserHistory(self,username):
        indices=np.where(self.names==username)[0]
        if len(indices)==0:
            return (np.array([]),np.array([]),np.array([]),np.array([]),np.array([]))

        ids=self.taskids[indices]
        subnum=self.subnums[indices]
        date=self.subdates[indices]
        score=self.scores[indices]
        rank=self.finalranks[indices]
        return (ids,subnum,date,score,rank)


class UserData:
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

    def getSelUsers(self,usernames):
        names=[]
        tenure=[]
        skills=[]

        for i in range(len(self.name)):
            name=self.name[i]
            if name in usernames:
                names.append(name)
                tenure.append(self.memberage[i])
                skills.append(self.skills[i])

        return UserDataContainer(names,tenure,skills)

class Registration:
    def __init__(self,):
        self.loadData()

    def loadData(self,):
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
        self.regdate=np.array(self.regdate,dtype=np.int)

        print("reg data num=%d"%len(dataset),"users=%d"%len(set(self.username)),"tasks=%d"%len(set(self.taskid)))

    def getSelRegistration(self,tasktype,taskids):
        ids=[]
        names=[]
        dates=[]
        for i in range(len(self.taskid)):
            id=self.taskid[i]
            if id in taskids:
                ids.append(self.taskid[i])
                names.append(self.username[i])
                dates.append(self.regdate[i])

        return RegistrationDataContainer(tasktype=tasktype,taskids=ids,usernames=names,regdates=dates)

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
            if data[4] is not None and eval(data[4])>0:
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
                if abs(X[p][1])<0.5:
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


        print("sub data num=%d"%len(self.taskid),"users=%d"%len(set(self.username)),"tasks=%d"%len(set(self.taskid)))

    def getSelSubmission(self,tasktype,taskids):
        ids=[]
        names=[]
        dates=[]
        nums=[]
        scores=[]
        ranks=[]

        for i in range(len(self.taskid)):
            id=self.taskid[i]
            if id in taskids:
                ids.append(self.taskid[i])
                names.append(self.username[i])
                dates.append(self.subdate[i])
                nums.append(self.subnum[i])
                scores.append(self.score[i])
                ranks.append(self.finalrank[i])


        return SubmissionDataContainer(tasktype=tasktype,taskids=ids,usernames=names,
                                       subnums=nums,subdates=dates,scores=scores,finalranks=ranks)


class Tasks:
    def __init__(self,tasktype,begindate=3000):

        filepath="../data/TaskInstances/taskDataSet/"+tasktype+"-taskData.data"

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            for k in data.keys():
                data[k]=list(data[k])
                data[k].reverse()

            self.taskIDs = data["ids"]
            self.docX=data["docX"]
            self.lans=data["lans"]
            self.techs=data["techs"]
            self.diffdegs=data["diffdegs"]
            self.postingdate=data["startdates"]
            self.durations=data["durations"]
            self.prizes=data["prizes"]

            print(tasktype,"=>","Task Instances data size=%d"%(len(self.taskIDs)))
            #print("post",self.postingdate[:20])
        taskdata = {}
        for i in range(len(self.taskIDs)):
            taskdata[self.taskIDs[i]]=i

        self.dataIndex=taskdata

        self.tasktype=tasktype
        self.taskdata=taskdata

        self.loadData(begindate)

    def loadData(self,begindate=4200):
        pos=0
        for pos in range(len(self.taskIDs)):
            if self.postingdate[pos]>=begindate:
                break

        self.taskIDs = self.taskIDs[:pos]
        self.docX=self.docX[:pos]
        self.lans=self.lans[:pos]
        self.techs=self.techs[:pos]
        self.diffdegs=self.diffdegs[:pos]
        self.postingdate=self.postingdate[:pos]
        self.durations=self.durations[:pos]
        self.prizes=self.prizes[:pos]

        print(self.tasktype+": task size=%d"%len(self.taskIDs))

    def ClipRatio(self,ratio=0.4):
        clip_pos=int(ratio*len(self.taskIDs))
        self.taskIDs=self.taskIDs[:clip_pos]
        self.docX=self.docX[:clip_pos]
        self.lans=self.lans[:clip_pos]
        self.techs=self.techs[:clip_pos]
        self.diffdegs=self.diffdegs[:clip_pos]
        self.postingdate=self.postingdate[:clip_pos]
        self.durations=self.durations[:clip_pos]
        self.prizes=self.prizes[:clip_pos]

class UserHistoryGenerator:
    def __init__(self,testMode=False):
        self.testMode=testMode
    def genActiveUserHistory(self,userdata,regdata,subdata,mode,tasktype,filepath=None):

        userhistory = {}

        for username in userdata.names:
            #fiter incomplete data information
            tenure,skills,skills_vec=userdata.getUserInfo(username)
            if tenure is None:
                continue

            regids, regdates = regdata.getUserHistory(username)
            if mode==0 and len(regids) <minRegNum:
                #default for those have registered
                continue
            subids, subnum, subdates, score, rank = subdata.getUserHistory(username)
            winindices=np.where(rank==0)[0]
            if mode==1 and np.sum(subnum)<minSubNum:
                #for those ever submitted
                continue
            if mode==2 and len(winindices)<minWinNum:
                #for those ever won
                continue

            regids=list(regids)
            regids.reverse()
            regdates=list(regdates)
            regdates.reverse()

            subids=list(subids)
            subids.reverse()
            subnum=list(subnum)
            subnum.reverse()
            subdates=list(subdates)
            subdates.reverse()
            score=list(score)
            score.reverse()
            rank=list(rank)
            rank.reverse()

            userhistory[username] = {"regtasks": [regids, regdates],
                                  "subtasks": [subids, subnum, subdates, score, rank],
                                  "tenure":tenure,"skills":skills.split(","),"skills_vec":skills_vec}

            #print(regdates)
            #print(subdates)

            #print(username, "sub histroy and reg histrory=", len(userData[username]["subtasks"][0]),
            #      len(userData[username]["regtasks"][0]))
        if self.testMode:
            tasktype=tasktype+"-test"
        if filepath is None:
            filepath="../data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data"
        print("saving %s history of %d users"%(ModeTag[mode], len(userhistory)),"type="+tasktype)
        with open(filepath, "wb") as f:
            pickle.dump(userhistory, f,True)

    def loadActiveUserHistory(self,tasktype,mode,filepath=None):
        print("loading %s history of active users "%ModeTag[mode])
        if self.testMode:
            tasktype=tasktype+"-test"

        if filepath is None:
            filepath="../data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data"
        with open(filepath, "rb") as f:
            userData = pickle.load(f)

        return userData

#test data
import matplotlib.pyplot as plt
from Utility.personalizedSort import *
if __name__ == '__main__':

    '''
    Users=UserData()
    Users=Users.getSelUsers(Users.name[np.where(Users.memberage>600)])
    y=Users.tenure.reshape((len(Users.tenure),1)).tolist()
    ms=MySort(y)
    y=ms.mergeSort()
    x=np.arange(len(y))
    plt.plot(x,y)
    plt.show()
    '''

