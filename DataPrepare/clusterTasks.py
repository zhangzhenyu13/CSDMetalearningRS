from DataPrepare.ConnectDB import *
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
import pickle,multiprocessing
from DataPrepare.DataContainer import TaskDataContainer
from ML_Models.ClusteringModel import ClusteringModel
from Utility.TagsDef import *
warnings.filterwarnings("ignore")

def showData(X):
    import Utility.personalizedSort as ps
    m_s=ps.MySort(X)
    m_s.compare_vec_index=-1
    y=m_s.mergeSort()
    x=np.arange(len(y))
    plt.plot(x,np.array(y)[:,0])
    plt.show()

def initDataSet():
        conn = ConnectDB()
        cur = conn.cursor()
        sqlcmd = 'select taskid,detail,taskname, duration,technology,languages,prize,postingdate,diffdeg,tasktype from task ' \
                 ' order by postingdate desc'
        cur.execute(sqlcmd)
        dataset = cur.fetchall()
        ids=[]
        docs=[]
        techs=[]
        lans=[]
        startdates=[]
        durations=[]
        prizes=[]
        diffdegs=[]
        tasktypes=[]

        for data in dataset:
            #print(data)
            if data[1] is None:
                continue
            ids.append(data[0])
            docs.append(data[2]+"\n"+data[1])

            if data[3]>50:
                durations.append([50])
            elif data[3]<1:
                durations.append([1])
            else:
                durations.append([data[3]])

            techs.append(data[4])
            lans.append(data[5])

            if data[6]!='':
                prize=np.sum(eval(data[6]))
                if prize>6000:
                    prize=6000
                if prize<1:
                    prize=1
                prizes.append([prize])
            else:
                prizes.append([1.])

            if data[7]<1:
                startdates.append([1])
            else:
                startdates.append([data[7]])

            if data[8]>0.6:
                diffdegs.append([0.6])
            else:
                diffdegs.append([data[8]])

            tasktypes.append(data[9])

        print("task size=",len(ids),len(docs),len(techs),len(lans),len(startdates),len(durations),len(prizes),len(diffdegs),len(tasktypes))

        #adding to corresponding type

        dataSet={}
        for i in range(len(tasktypes)):
            t=tasktypes[i]
            if t is None:
                continue
            if t in dataSet.keys():

                container=dataSet[t]

                container.ids.append(ids[i])
                container.docs.append(docs[i])
                container.techs.append(techs[i])
                container.lans.append(lans[i])
                container.startdates.append(startdates[i])
                container.durations.append(durations[i])
                container.prizes.append(prizes[i])
                container.diffdegs.append(diffdegs[i])

                dataSet[t]=container

            else:

                container=TaskDataContainer(typename=t)

                container.ids.append(ids[i])
                container.docs.append(docs[i])
                container.techs.append(techs[i])
                container.lans.append(lans[i])
                container.startdates.append(startdates[i])
                container.durations.append(durations[i])
                container.prizes.append(prizes[i])
                container.diffdegs.append(diffdegs[i])

                dataSet[t]=container

        typeInfo={}
        selTypeInfo={}
        for t in dataSet.keys():
            typeInfo[t]=dataSet[t].ids
            if len(dataSet[t].ids)>TaskFilterThreshold:
                selTypeInfo[t]=dataSet[t].ids

        with open("../data/TaskInstances/OriginalTasktype.data","wb") as f:
            pickle.dump(typeInfo,f)

        with open("../data/TaskInstances/SelTasktype.data","wb") as f:
            pickle.dump(selTypeInfo,f)

        with open("../data/TaskInstances/OriginalTasktype.data", "rb") as f:
            typeInfo=pickle.load(f)
            total=0
            for k in typeInfo.keys():print(k,",",len(typeInfo[k]));total+=len(typeInfo[k])
        print("original task num=%d of %d" %(total,len(ids)))
        print()

        print("Selected Task Item Type")
        with open("../data/TaskInstances/SelTasktype.data", "rb") as f:
            selTypeInfo=pickle.load(f)
            total=0
            for k in selTypeInfo.keys():print(k,",",len(selTypeInfo[k]));total+=len(selTypeInfo[k])
        print("Selected task num=%d of %d" %(total,len(ids)))
        return dataSet

def clusterVec(taskdata,docX):

    X_techs=taskdata.techs_vec
    X_lans=taskdata.lans_vec

    print("cluster shape: docs,techs,lans",docX.shape,X_techs.shape,X_lans.shape)
    X=np.concatenate((docX,X_techs),axis=1)
    X=np.concatenate((X,X_lans),axis=1)

    return X


#save data content as a vector
def saveTaskData(taskdata):
    data={}
    docX=taskdata.docs
    oldshape=docX.shape
    kpca=decomposition.KernelPCA(n_components=min(len(docX[0]),100),kernel="rbf")
    data["docX"]=kpca.fit_transform(docX)
    print(taskdata.taskType,"KPCA doc shape changed from",oldshape,"to",data["docX"].shape)

    #print(taskdata.techs[:20])
    #print(taskdata.lans[:20])
    #exit(20)
    for i in range(len(taskdata.ids)):
        taskdata.lans[i]=taskdata.lans[i].split(",")
        taskdata.techs[i]=taskdata.techs[i].split(",")
    #print(taskdata.techs[:20])
    #print(taskdata.lans[:20])
    #exit(20)
    data["lans"]=taskdata.lans
    data["techs"]=taskdata.techs
    data["diffdegs"]=taskdata.diffdegs
    data["startdates"]=taskdata.startdates
    data["durations"]=taskdata.durations
    data["prizes"]=taskdata.prizes
    data["ids"]=taskdata.ids
    with open("../data/TaskInstances/taskDataSet/"+taskdata.taskType+"-taskData.data","wb") as f:
        pickle.dump(data,f)


def genResultOfTasktype(tasktype,taskdata,choice):

    taskdata.encodingFeature(choice)
    docX=taskdata.docs

    #save vec representaion of all the tasks
    saveTaskData(taskdata)

    #cluster task based on its feature
    X=clusterVec(taskdata,docX)

    clusterEXP=800
    model=ClusteringModel()
    model.name=tasktype+"-clusteringModel"
    n_clusters=max(1,len(taskdata.ids)//clusterEXP)

    model.trainCluster(X=X,n_clusters=n_clusters,minibatch=False)

    model.saveModel()
    model.loadModel()
    result=model.predictCluster(X)

    IDClusters={}
    for i in range(n_clusters):
        IDClusters[i]=[]
    for i in range(len(result)):
        t=result[i]

        if t not in IDClusters.keys():
            IDClusters[t]=[taskdata.ids[i]]
        else:
            IDClusters[t].append(taskdata.ids[i])

    # saving result
    print("saving clustering result")
    with open("../data/TaskInstances/taskClusterSet/"+tasktype+"-clusters.data", "wb") as f:
        pickle.dump(IDClusters, f)

    with open("../data/TaskInstances/taskClusterSet/"+tasktype+"-clusters.data", "rb") as f:
        taskidClusters=pickle.load(f)

    print("saving cluster plot result")
    plt.figure(tasktype)
    y=[]
    for i in range(n_clusters):
        y.append(len(taskidClusters[i]))
        print(tasktype,"#%d"%i,"size=%d"%len(taskidClusters[i]))

    plt.plot(np.arange(n_clusters),y, marker='o')
    plt.title(tasktype+", size=%d"%len(X))
    plt.xlabel("cluster no")
    plt.ylabel("task instance size")
    plt.savefig("../data/pictures/TaskClusterPlots/"+tasktype+ "-taskclusters.png")
    plt.gcf().clear()
    print("===========================================================================")
    print()

def genResults():
    dataSet=initDataSet()
    typeinfo=dataSet.keys()

    choice=eval(input("1:LDA; 2:LSA \t"))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    with open("../data/TaskInstances/SelTasktype.data","rb") as f:
        tasktypes=pickle.load(f)
    for t in tasktypes:
        taskdata=dataSet[t]
        tasktype=taskdata.taskType

        #print(taskdata.ids);exit(10)
        multiprocessing.Process(target=genResultOfTasktype,args=(tasktype,taskdata,choice)).start()

        #genResultOfTasktype(tasktype=tasktype,taskdata=taskdata,choice=choice)



if __name__ == '__main__':
    genResults()

