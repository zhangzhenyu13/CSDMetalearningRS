ModeTag={0:"Reg",1:"Sub",2:"Win"}
TestDate=600
TaskFeatures=100
TaskLans=18
TaskTechs=100
UserSkills=100

import pickle
def getUsers(tasktype,mode):
    with open("../data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data","rb") as f:
        data=pickle.load(f)
    return list(data.keys())
