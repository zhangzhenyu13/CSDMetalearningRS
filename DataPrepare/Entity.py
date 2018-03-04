from DataPrepare.ConnectDB import *
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
            self.memberage.append(data[1])
            self.skills.append(data[2])
            self.competitionNum.append(data[3])
            self.submissionNum.append(data[4])
            self.winNum.append(data[5])

        print("users num=%d"%len(self.name))
        #construct skill dict
        skillset=set()
        for skill in self.skills:
            if skill is None or skill=='':
                continue
            skills=skill.split(",")
            for sk in skills:
                skillset.add(sk)
        skillset=list(skillset)
        self.skilldict={}

        for i in range(len(skillset)):
            self.skilldict[skillset[i]]=i

    def skilltovec(self,skilldata):
        X=[]
        for index in range(len(skilldata)):
            skill=skilldata[index]
            skillVec=np.zeros(shape=len(self.skilldict),dtype=np.float32)

            if skill is not None and skill!='':
                skills=skill.split(",")
                for sk in skills:
                    i=self.skilldict[sk]
                    skillVec[i]=1
            X.append(skillVec)
        return np.array(X)

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
        sqlcmd = 'select taskid, handle,regdate from registration'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.taskid=[]
        self.username=[]
        self.regdate=[]
        for data in dataset:
            self.taskid.append(data[0])
            self.username.append(data[1])
            self.regdate.append(data[2])
        print("registration num=%d"%len(dataset))

class Submission:
    def __init__(self):
        self.loadData()
    def loadData(self):
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid,handle,subnum,score from submission'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        self.taskid=[]
        self.username=[]
        self.subnum=[]
        self.score=[]
        for data in dataset:
            self.taskid.append(data[0])
            self.username.append(data[1])
            self.subnum.append(data[2])
            self.score.append(data[3])
        print("sub num=%d"%len(self.taskid))

if __name__ == '__main__':
    user=Users()

    for i in range(10):
        print(user.name[i],user.skills[i],user.memberage[i],user.competitionNum[i],user.submissionNum[i],user.winNum[i])

    user.skills = user.skilltovec(user.skills)
    X = user.transformVec()
    print(len(user.skilldict))
    print(user.skilldict)
    print(X.shape)
    for x in X[:10]:
        print(x)

    regs=Registration()
    for i in range(10):
        id=regs.taskid[i]
        name=regs.username[i]
        date=regs.regdate[i]
        print(id,name,date)

    subs=Submission()
    for i in range(10):
        id=subs.taskid[i]
        name=subs.username[i]
        num=subs.subnum[i]
        score=subs.score[i]
        print(id,name,num,score)
