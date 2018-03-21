import pickle
import os
import numpy as np
import gc
from numpy import linalg as LA
def testReg():
    with open("/home/bird/Desktop/regsdata/task_userReg1.data0_165","rb") as f:
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
    with open("../data/Instances/task_user1.data","rb") as f:
        data=pickle.load(f)

        taskids=data["taskids"]
        #tasks=data["tasks"]
        #users=data["users"]
        #subdates=data["dates"]
        sub=data["submits"]
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

if __name__ == '__main__':
    testSub()
    #testReg()