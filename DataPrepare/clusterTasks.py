from DataPrepare.ConnectDB import *
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
import pickle,multiprocessing
from DataPrepare.DataContainer import TaskDataContainer
from ML_Models.ClusteringModel import ClusteringModel

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

        print()
        typeInfo={}
        for t in dataSet.keys():
            typeInfo[t]=dataSet[t].ids
        with open("../data/TaskInstances/OriginalTasktype.data","wb") as f:
            pickle.dump(typeInfo,f)
        with open("../data/TaskInstances/OriginalTasktype.data", "rb") as f:
            typeInfo=pickle.load(f)
            total=0
            for k in typeInfo.keys():print(k,",",len(typeInfo[k]));total+=len(typeInfo[k])

        print()
        print("task num=%d" % len(ids),total)

        return dataSet

def clusterVec(taskdata,docX):

    X_techs=taskdata.techs
    X_lans=taskdata.lans

    print("docs,techs,lans",docX.shape,X_techs.shape,X_lans.shape)
    X=np.concatenate((docX,X_techs),axis=1)
    X=np.concatenate((X,X_lans),axis=1)

    print("cluster shape doc",docX.shape,"vec shape",X.shape)

    return X

def taskVec(taskdata,docX):

    print("doc shape",docX.shape)
    kpca=decomposition.KernelPCA(n_components=100,kernel="rbf")
    docX=kpca.fit_transform(docX)
    print("doc shape changed to",docX.shape)

    X_techs=taskdata.techs
    X_lans=taskdata.lans

    X_startdate = taskdata.startdates
    X_duration = taskdata.durations
    X_prize = taskdata.prizes
    X_diffdeg =taskdata.diffdegs

    X = np.concatenate((docX,X_techs), axis=1)
    X = np.concatenate((X, X_lans), axis=1)
    X = np.concatenate((X, X_startdate), axis=1)
    X = np.concatenate((X, X_duration), axis=1)
    X = np.concatenate((X, X_prize), axis=1)
    X = np.concatenate((X, X_diffdeg), axis=1)
    return X

#save data content as a vector
def saveTaskVecData(X,taskids,dataname,choice=1):
    data={}
    data["size"]=len(taskids)
    data["taskids"]=taskids
    data["tasks"]=X
    with open("../data/TaskInstances/taskDataSet/"+dataname+"-taskData-"+str(choice)+".data","wb") as f:
        pickle.dump(data,f)


def genResultOfTasktype(tasktype,taskdata,choice):

    taskdata=taskdata
    taskdata.encodingFeature(choice)
    docX=taskdata.docs
    #kernel pca method test
    '''
    print()
    print(tasktype,docX.shape,"before kPCA")
    kpca=decomposition.KernelPCA()
    docX=kpca.fit_transform(docX)
    print(tasktype,docX.shape,"after kPCA")
    print(tasktype,kpca.coef0)
    print()
    '''

    #save vec representaion of all the tasks
    X=taskVec(taskdata,docX)
    saveTaskVecData(X,taskdata.ids,tasktype,choice)

    #cluster task based on its feature
    X=clusterVec(taskdata,docX)

    clusterEXP=500
    model=ClusteringModel()
    model.name=tasktype+"-clusteringModel-"+str(choice)
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
    with open("../data/TaskInstances/taskClusterSet/"+tasktype+"-clusters-" + str(choice) + ".data", "wb") as f:
        pickle.dump(IDClusters, f)

    with open("../data/TaskInstances/taskClusterSet/"+tasktype+"-clusters-"  + str(choice) + ".data", "rb") as f:
        taskidClusters=pickle.load(f)

    print("saving cluster plot result")
    plt.figure(tasktype)
    y=[]
    for i in range(n_clusters):
        y.append(len(taskidClusters[i]))
        print(tasktype,"#%d"%i,"size=%d"%len(taskidClusters[i]))

    plt.plot(np.arange(n_clusters),y, marker='o')
    plt.title(tasktype+",choice=%d" % choice+", size=%d"%len(X))
    plt.xlabel("cluster no")
    plt.ylabel("task instance size")
    plt.savefig("../data/pictures/TaskClusterPlots/"+tasktype+ "-taskclusters-"+str(choice)+".png")
    plt.gcf().clear()
    print("===========================================================================")
    print()

def genResults():
    dataSet=initDataSet()
    typeinfo=dataSet.keys()

    choice=eval(input("1:LDA; 2:LSA \t"))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    filterThreshold=100
    for t in typeinfo:
        taskdata=dataSet[t]
        tasktype=taskdata.taskType
        if len(taskdata.ids)<filterThreshold:
            continue

        #print(taskdata.ids);exit(10)
        multiprocessing.Process(target=genResultOfTasktype,args=(tasktype,taskdata,choice)).start()

        #genResultOfTasktype(tasktype=tasktype,taskdata=taskdata,choice=choice)



if __name__ == '__main__':
    genResults()

