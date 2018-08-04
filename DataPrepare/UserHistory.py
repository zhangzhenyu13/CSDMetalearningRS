import multiprocessing
from DataPrepare.DataContainer import *
from Utility import SelectedTaskTypes
import _pickle as pickle
warnings.filterwarnings("ignore")


def genUserHistoryOfTaskType(userhistory,tasktype,Users,Regs,Subs):
    with open("../data/TaskInstances/taskDataSet/"+tasktype+"-taskData.data","rb") as f:
        taskdata=pickle.load(f)
    taskids=taskdata["ids"]

    regdata=Regs.getSelRegistration(tasktype=tasktype,taskids=taskids)
    subdata=Subs.getSelSubmission(tasktype=tasktype,taskids=taskids)
    selnames=regdata.getAllUsers()

    userdata=(Users.getSelUsers(usernames=selnames))
    with open("../data/UserInstances/SkillEncoding.data","rb") as f:
        skills_feature=pickle.load(f)

    userdata.skills_vec=EncodeByDict(userdata.skills,skills_feature,UserSkills)

    for i in range(len(userdata.names)):
        if userdata.skills[i] is None:
            userdata.skills[i]=""

    with open("../data/TaskInstances/RegInfo/"+tasktype+"-regs.data","wb") as f:
        data={}
        data["taskids"]=regdata.taskids
        data["regdates"]=regdata.regdates
        data["names"]=regdata.names
        pickle.dump(data,f,True)
        print("saved %d reg items"%len(regdata.taskids))

    with open("../data/TaskInstances/SubInfo/"+tasktype+"-subs.data","wb") as f:
        data={}
        data["taskids"]=subdata.taskids
        data["subdates"]=subdata.subdates
        data["names"]=subdata.names
        data["subnums"]=subdata.subnums
        data["scores"]=subdata.scores
        data["finalranks"]=subdata.finalranks
        pickle.dump(data,f,True)
        print("saved %d sub items"%len(subdata.taskids))

    #return
    for mode in (2,):
        userhistory.genActiveUserHistory(userdata=userdata,regdata=regdata,subdata=subdata,mode=mode,tasktype=tasktype)

def skillEncoding():
    #skill feature_dict
    print("saving skills features")
    with open("../data/TaskInstances/GlobalEncoding.data","rb") as f:
        data=pickle.load(f)
        taskids=data["ids"]
    gsubdata=Subs.getSelSubmission("global",taskids)
    allsubusers=gsubdata.getAllUsers()
    allsubskills=[]
    for i in range(len(Users.name)):
        name=Users.name[i]
        if name in allsubusers:
            allsubskills.append(Users.skills[i])
    skills_feature=onehotFeatures(allsubskills)
    print(skills_feature)
    with open("../data/UserInstances/SkillEncoding.data","wb") as f:
        pickle.dump(skills_feature,f,True)

if __name__ == '__main__':
    #init data set
    Regs=Registration()
    Subs=Submission()
    Users=UserData()
    #
    #skillEncoding()
    userhistory=UserHistoryGenerator()
    userhistory.testMode=True
    #construct global users set
    #genUserHistoryOfTaskType(userhistory,"global",Users,Regs,Subs);exit(10)
    #construct history for users of given tasktype

    tasktypes=SelectedTaskTypes.loadTaskTypes()
    for t in tasktypes["keeped"]:
        #genUserHistoryOfTaskType(userhistory=userhistory,tasktype=t,Users=Users,Regs=Regs,Subs=Subs)
        multiprocessing.Process(target=genUserHistoryOfTaskType,args=(userhistory,t,Users,Regs,Subs)).start()

