ModeTag={0:"Reg",1:"Sub",2:"Win"}
TestDate=600
TaskFeatures=100
TaskLans=18
TaskTechs=100
UserSkills=100
#reg, sub, win :threshold
minRegNum=30
minSubNum=10
minWinNum=5

import pickle
def getUsers(tasktype,mode=2):
    with open("../data/UserInstances/"+tasktype+"-Users"+ModeTag[mode]+".data","rb") as f:
        usersList=pickle.load(f)
    return usersList

def genSelectedUserlist(tasktype,mode=2):
    with open("../data/UserInstances/UserHistory/"+tasktype+"-UserHistory"+ModeTag[mode]+".data","rb") as f:
        data=pickle.load(f)
    usersList=list(data.keys())

    with open("../data/UserInstances/"+tasktype+"-Users"+ModeTag[mode]+".data","wb") as f:
        pickle.dump(usersList,f)

if __name__ == '__main__':
    from Utility.SelectedTaskTypes import loadTaskTypes
    tasltypes=loadTaskTypes()
    mode=0
    for t in tasltypes["keeped"]:
        genSelectedUserlist(t,mode)
        print(getUsers(t,mode))
