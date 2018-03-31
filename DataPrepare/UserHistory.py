from DataPrepare.ConnectDB import *
import multiprocessing,threading
from DataPrepare.DataContainer import *
warnings.filterwarnings("ignore")


def genUserHistoryOfTaskType(userhistory,tasktype,Users,Regs,Subs):
    with open("../data/TaskInstances/taskDataSet/"+t+"-taskData.data","rb") as f:
        taskdata=pickle.load(f)
    taskids=taskdata["ids"]

    regdata=Regs.getSelRegistration(tasktype=tasktype,taskids=taskids)
    subdata=Subs.getSelSubmission(tasktype=tasktype,taskids=taskids)
    selnames=regdata.getAllUsers()
    userdata=(Users.getSelUsers(usernames=selnames))
    userdata.skills_vec=onehotFeatures(data=userdata.skills,threshold_num=0.5*len(userdata.names))

    for i in range(len(userdata.names)):
        if userdata.skills[i] is None:
            userdata.skills[i]=""

    with open("../data/TaskInstances/RegInfo/"+tasktype+"-regs.data","wb") as f:
        data={}
        data["taskids"]=regdata.taskids
        data["regdates"]=regdata.regdates
        data["names"]=regdata.names
        pickle.dump(data,f)
        print("saved %d reg items"%len(regdata.taskids))

    with open("../data/TaskInstances/SubInfo/"+tasktype+"-subs.data","wb") as f:
        data={}
        data["taskids"]=subdata.taskids
        data["subdates"]=subdata.subdates
        data["names"]=subdata.names
        data["subnums"]=subdata.subnums
        data["scores"]=subdata.scores
        data["finalranks"]=subdata.finalranks
        pickle.dump(data,f)
        print("saved %d sub items"%len(subdata.taskids))

    #return
    for mode in (0,1,2):
        userhistory.genActiveUserHistory(userdata=userdata,regdata=regdata,subdata=subdata,mode=mode,tasktype=tasktype)


if __name__ == '__main__':
    #init data set
    Regs=Registration()
    Subs=Submission()
    Users=UserData()

    userhistory=UserHistoryGenerator()
    #construct history for users of given tasktype

    with open("../data/TaskInstances/TaskIndex.data","rb") as f:
        tasktypes=pickle.load(f)

    for t in tasktypes:

        #genUserHistoryOfTaskType(userhistory=userhistory,tasktype=t,Users=Users,Regs=Regs,Subs=Subs)
        multiprocessing.Process(target=genUserHistoryOfTaskType,args=(userhistory,t,Users,Regs,Subs)).start()


