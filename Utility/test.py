import pickle
import os
import numpy as np
from Utility import SelectedTaskTypes
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
    tasktypes=SelectedTaskTypes.loadTaskTypes()
    for t in tasktypes["keeped"]:
        count+=1
        if t !="Architecture":continue
        for mode in (2,):
            userdata=userhis.loadActiveUserHistory(tasktype=t,mode=mode)
            print(count,t,ModeTag[mode]+":%d"%len(userdata))
            print(list(userdata.keys()))
        print()


if __name__ == '__main__':
    #testSub()
    #testReg()
    #scanID()
    #testFileIndex()
    countUsers()
    countUsers()
    countUsers()
