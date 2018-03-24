from DataPrepare.ConnectDB import *

import multiprocessing
from DataPrepare.DataContainer import *
warnings.filterwarnings("ignore")


def genUserHistoryOfTaskType(taskids,userhistory,tasktype,Users,Regs,Subs):

    regdata=Regs.getSelRegistration(tasktype=tasktype,taskids=taskids)
    subdata=Subs.getSelSubmission(tasktype=tasktype,taskids=taskids)
    selnames=regdata.getAllUsers()
    userdata=Users.getSelUsers(usernames=selnames)
    userdata.skills=onehotFeatures(userdata.skills)

    with open("../data/TaskInstances/RegInfo/"+tasktype+"-regs-"+str(choice)+".data","wb") as f:
        data={}
        data["taskids"]=regdata.taskids
        data["regdates"]=regdata.regdates
        data["names"]=regdata.names
        pickle.dump(data,f)
        print("saved %d reg items"%len(regdata.taskids))

    with open("../data/TaskInstances/SubInfo/"+tasktype+"-subs-"+str(choice)+".data","wb") as f:
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
    choice=1
    Regs=Registration()
    Subs=Submission()
    Users=UserData()
    userhistory=UserHistoryGenerator()
    #construct history for users of given tasktype
    with open("../data/TaskInstances/OriginalTasktype.data","rb") as f:
        tasktypes=pickle.load(f)
        for t in tasktypes.keys():
            if len(tasktypes[t])<50:
                continue

            taskids=tasktypes[t]
            tasktype=t.replace("/","_")

            #genUserHistoryOfTaskType(taskids=taskids,userhistory=userhistory,tasktype=tasktype,Users=Users,Regs=Regs,Subs=Subs)
            multiprocessing.Process(target=genUserHistoryOfTaskType,args=(taskids,userhistory,tasktype,Users,Regs,Subs)).start()


