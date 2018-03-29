import pickle
import os
import numpy as np
import gc
from numpy import linalg as LA
import matplotlib.pyplot as plt
from Utility.TagsDef import *
openMode="rb"
def testReg():
    with open("../data/Instances/regsdata/task_userReg2.data0","rb") as f:
        data=pickle.load(f)

        taskids=data["taskids"]
        tasks=data["tasks"]
        users=data["users"]
        regdates=data["dates"]
        regists=data["regists"]
    id =taskids[0]
    p=0
    i=0
    postiveI=[]
    negativeI=[]
    while i+p<len(taskids):
        curID=taskids[p+i]
        if curID!=id:
            print(id,len(negativeI),len(postiveI))
            if len(negativeI)>1.5*len(postiveI):
                pass
            i+=p
            p=0
            id=taskids[i]
            postiveI=[]
            negativeI=[]
        if regists[p+i]==1:
            postiveI.append(p+i)
        else:
            negativeI.append(p+i)

        p+=1

def testSub():
    with open("../data/Instances/subsdata/task_user1.data","rb") as f:
        data=pickle.load(f)

        taskids=data["taskids"]
        #tasks=data["tasks"]
        #users=data["users"]
        #subdates=data["dates"]
        sub=data["submits"]
        print(taskids[1000:1020])
        print(sub[1000:1020])
        plt.plot(np.arange(len(sub)),sub)
        plt.show()
        #exit()
    id =taskids[0]
    p=0
    i=0
    postiveI=[]
    negativeI=[]
    while i+p<len(taskids):
        curID=taskids[p+i]
        if curID!=id:
            print(id,len(negativeI),len(postiveI))
            if len(negativeI)>1.5*len(postiveI):
                pass
            i+=p
            p=0
            id=taskids[i]
            postiveI=[]
            negativeI=[]
        if sub[p+i]>0:
            postiveI.append(p+i)
        else:
            negativeI.append(p+i)

        p+=1
def scanID():
    with open("../data/clusterResult/clusters2.data", "rb") as f:
        taskidClusters=pickle.load(f)
        #print(taskidClusters['First2Finish2'])
        for k in range(6):
            s="select * from task where"
            for id in taskidClusters["First2Finish"+str(k)]:
                s=s+" taskid='"+str(id)+"' or"
            s=s[:-3]+";"
            print(s)


def testFileIndex():
    with open("../data/TopcoderDataSet/subHistoryBasedData/Assembly Competition-user_task-1.data","rb") as f:
        files=pickle.load(f)
        for file in files:
            print(file)
def countUsers():
    from DataPrepare.DataContainer import UserHistoryGenerator
    userhis=UserHistoryGenerator()
    count=0
    with open("../data/TaskInstances/OriginalTasktype.data","rb") as f:
        tasktypes=pickle.load(f)
        for t in tasktypes.keys():
            count+=1
            tasktype=t.replace("/","_")
            for mode in (0,1,2):
                userdata=userhis.loadActiveUserHistory(tasktype=tasktype,mode=mode)
                print(count,tasktype,ModeTag[mode]+":%d"%len(userdata))
            print()
def countUserTaskType():
    from DataPrepare.DataContainer import UserHistoryGenerator
    userhis=UserHistoryGenerator()
    usertypesReg={}
    usertypesSub={}
    usertypesWin={}

    with open("../data/TaskInstances/OriginalTasktype.data","rb") as f:
        tasktypeIndex=pickle.load(f)
        tasktypes=list(tasktypeIndex.keys())
        for i in range(len(tasktypes)):

            tasktype=tasktypes[i].replace("/","_")

            userdata=userhis.loadActiveUserHistory(tasktype=tasktype,mode=0)
            usertypesReg[tasktype]=set(userdata.keys())
            #print(type(usertypesReg[tasktype]),usertypesReg[tasktype]);exit(10)
            userdata=userhis.loadActiveUserHistory(tasktype=tasktype,mode=1)
            usertypesSub[tasktype]=set(userdata.keys())

            userdata=userhis.loadActiveUserHistory(tasktype=tasktype,mode=2)
            usertypesWin[tasktype]=set(userdata.keys())

        userCrossTypeData={}
        userCrossTypeData["tasktypes"]=tasktypes
        crossM_Reg={}
        crossM_Sub={}
        crossM_Win={}

        x=np.arange(len(tasktypes))

        for i in range(len(tasktypes)):
            t1=tasktypes[i].replace("/","_")

            for j in range(i+1,len(tasktypes)):

                t2=tasktypes[j].replace("/","_")

                comTReg=usertypesReg[t1].intersection(usertypesReg[t2])
                comTSub=usertypesSub[t1].intersection(usertypesSub[t2])
                comTWin=usertypesWin[t1].intersection(usertypesWin[t2])

                crossM_Reg[(t1,t2)]=crossM_Reg[(t2,t1)]=comTReg
                crossM_Sub[(t1,t2)]=crossM_Sub[(t2,t1)]=comTSub
                crossM_Win[(t1,t2)]=crossM_Win[(t2,t1)]=comTWin



                print("between %s and %s"%(t1,t2))
                print("regs type common=%d"%len(comTReg))
                print("subs type common=%d"%len(comTSub))
                print("wins type common=%d"%len(comTWin))
                print()

        userCrossTypeData["regs"]=crossM_Reg
        userCrossTypeData["subs"]=crossM_Sub
        userCrossTypeData["wins"]=crossM_Win
        with open("../data/Statistics/crossTypeUserData.data","wb") as f:
            pickle.dump(userCrossTypeData,f)

        with open("../data/Statistics/crossTypeUserData.data","rb") as f:
            userCrossTypeData=pickle.load(f)
            crossM_Reg=userCrossTypeData["regs"]
            crossM_Sub=userCrossTypeData["subs"]
            crossM_Win=userCrossTypeData["wins"]

            for i in range(len(tasktypes)):
                t1=tasktypes[i].replace("/","_")
                plt.figure(t1)
                y1=np.zeros(shape=len(tasktypes))
                y2=np.zeros(shape=len(tasktypes))
                y3=np.zeros(shape=len(tasktypes))

                for j in range(len(tasktypes)):
                    if i==j:
                        continue
                    t2=tasktypes[j].replace("/","_")
                    y1[j]=len(crossM_Reg[(t1,t2)])
                    y2[j]=len(crossM_Sub[(t1,t2)])
                    y3[j]=len(crossM_Win[(t1,t2)])

                plt.plot(x,y1,color="r")
                plt.plot(x,y2,color="g")
                plt.plot(x,y3,color="b")
                plt.xlabel("task type no")
                plt.ylabel("intersection user num")
                plt.title(t1+"=>itme size=%d, user"%len(tasktypeIndex[t1.replace("_","/")]))
                plt.text(1000,1000,"red:reg")
                plt.text(1000,980,"green:sub")
                plt.text(1000,960,"blue:win")
                plt.savefig("../data/pictures/userCrossTypes/"+t1+".png")
                plt.gca().clear()


if __name__ == '__main__':
    #testSub()
    #testReg()
    #scanID()
    #testFileIndex()
    #countUsers()
    countUserTaskType()
