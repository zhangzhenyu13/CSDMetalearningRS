ModeTag={0:"Reg",1:"Sub",2:"Win"}
TestDate=600
TaskFeatures=60
TaskLans=18
TaskTechs=100
UserSkills=50
#reg, sub, win :threshold
minRegNum=30
minSubNum=10
minWinNum=1

import _pickle as pickle

def getUsers(tasktype,mode=2):
    with open("../data/UserInstances/"+tasktype+"-Users"+ModeTag[mode]+".data","rb") as f:
        usersList=pickle.load(f)
    return usersList

def genSelectedUserlist(tasktype,mode=2):
    with open("../data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data","rb") as f:
        data=pickle.load(f)
    usersList=list(data.keys())

    with open("../data/UserInstances/"+tasktype+"-Users"+ModeTag[mode]+".data","wb") as f:
        pickle.dump(usersList,f,True)

if __name__ == '__main__':
    from Utility.SelectedTaskTypes import loadTaskTypes

    tasltypes=loadTaskTypes()
    testMode=False
    for mode in (2,):
        #genSelectedUserlist("global",mode);continue
        #for k in tasltypes.keys():
            for t in tasltypes["keeped"]:
                if testMode:
                    t=t+"-test"
                genSelectedUserlist(t,mode)
                print(t,len(getUsers(t,mode)))
