import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from Utility import SelectedTaskTypes
def genCRTable(tasktypes):
    from DataPrepare.DataContainer import UserHistoryGenerator
    userhis=UserHistoryGenerator()
    usertypesReg={}
    usertypesSub={}
    usertypesWin={}

    #print(len(tasktypes),tasktypes);exit(10)

    for i in range(len(tasktypes)):

        tasktype=tasktypes[i]

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
        t1=tasktypes[i]

        for j in range(len(tasktypes)):
            if i==j:
                continue
            t2=tasktypes[j]

            if j>i:

                comTReg=usertypesReg[t1].intersection(usertypesReg[t2])
                comTSub=usertypesSub[t1].intersection(usertypesSub[t2])
                comTWin=usertypesWin[t1].intersection(usertypesWin[t2])

                crossM_Reg[(t1,t2)]=crossM_Reg[(t2,t1)]=comTReg
                crossM_Sub[(t1,t2)]=crossM_Sub[(t2,t1)]=comTSub
                crossM_Win[(t1,t2)]=crossM_Win[(t2,t1)]=comTWin

            comTReg=crossM_Reg[(t1,t2)]
            comTSub=crossM_Sub[(t1,t2)]
            comTWin=crossM_Win[(t1,t2)]
            print("between %s and %s"%(t1,t2))
            print("regs type common=%d"%len(comTReg),len(usertypesReg[t1]))
            print("subs type common=%d"%len(comTSub),len(usertypesSub[t1]))
            print("wins type common=%d"%len(comTWin),len(usertypesWin[t1]))
            print()

    userCrossTypeData["regs"]=crossM_Reg
    userCrossTypeData["subs"]=crossM_Sub
    userCrossTypeData["wins"]=crossM_Win
    with open("../data/Statistics/crossTypeUserData.data","wb") as f:
        pickle.dump(userCrossTypeData,f,True)

    with open("../data/Statistics/crossTypeUserData.data","rb") as f:
        userCrossTypeData=pickle.load(f)
        crossM_Reg=userCrossTypeData["regs"]
        crossM_Sub=userCrossTypeData["subs"]
        crossM_Win=userCrossTypeData["wins"]

        CR_Matrix={}
        CR_Matrix["tasktypes"]=tasktypes

        for i in range(len(tasktypes)):
            t1=tasktypes[i]
            plt.figure(t1)
            CR={}
            y1=np.zeros(shape=len(tasktypes))
            y2=np.zeros(shape=len(tasktypes))
            y3=np.zeros(shape=len(tasktypes))

            for j in range(len(tasktypes)):
                if i==j:
                    continue

                t2=tasktypes[j]

                if len(usertypesReg[t1])>0:
                    y1[j]=len(crossM_Reg[(t1,t2)])/len(usertypesReg[t1])

                if len(usertypesSub[t1])>0:
                    y2[j]=len(crossM_Sub[(t1,t2)])/len(usertypesSub[t1])

                if len(usertypesWin[t1])>0:
                    y3[j]=len(crossM_Win[(t1,t2)])/len(usertypesWin[t1])

            CR["regs"]=y1
            CR["subs"]=y2
            CR["wins"]=y3
            CR_Matrix[t1]=CR

            plt.plot(x,y1,color="b")
            plt.plot(x,y2,color="g")
            plt.plot(x,y3,color="r")
            plt.xlabel("task type no")
            plt.ylabel("CR")
            #t1=t1.replace("_","/")
            plt.title(t1)

            #plt.text(20,0.8,"red:reg")
            #plt.text(20,0.75,"green:sub")
            #plt.text(20,0.7,"blue:win")
            plt.savefig("../data/pictures/userCrossTypes/"+t1+".png")
            plt.gca().clear()

        with open("../data/Statistics/CR_Data.data","wb") as f:
            pickle.dump(CR_Matrix,f,True)

def gentransferNeighbors(cr_threshold_reg=0.8,cr_threshold_sub=0.6,cr_threshold_win=0.4):

    with open("../data/Statistics/CR_Data.data","rb") as f:
        CR_Matrix=pickle.load(f)

    tasktypes=CR_Matrix["tasktypes"]

    for t in tasktypes:
        regs_neighbors=[]
        subs_neighbor=[]
        wins_neighbor=[]
        cr=CR_Matrix[t]
        cr_regs=cr["regs"]
        cr_subs=cr["subs"]
        cr_wins=cr["wins"]

        for i in range(len(tasktypes)):
            candidate_nighbor=tasktypes[i]
            if candidate_nighbor==t:
                continue
            if cr_regs[i]>=cr_threshold_reg:
                regs_neighbors.append((candidate_nighbor,cr_regs[i]))
            if cr_subs[i]>=cr_threshold_sub:
                subs_neighbor.append((candidate_nighbor,cr_subs[i]))
            if cr_wins[i]>=cr_threshold_win:
                wins_neighbor.append((candidate_nighbor,cr_wins[i]))
        print(t,"regs nb",regs_neighbors)
        print(t,"subs nb",subs_neighbor)
        print(t,"wins nb",wins_neighbor)
        print()

if __name__ == '__main__':
    tasktypes=SelectedTaskTypes.loadTaskTypes()
    genCRTable(tasktypes["keeped"])
    gentransferNeighbors(cr_threshold_win=0)
