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

        #filtering data

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
            typeInfo[t]=len(dataSet[t].ids)
        with open("../data/clusterResult/tasktypeCluster.data","wb") as f:
            pickle.dump(typeInfo,f)
        with open("../data/clusterResult/tasktypeCluster.data", "rb") as f:
            typeInfo=pickle.load(f)
            for k in typeInfo.keys():print(k,",",len(typeInfo[k]))

        print()
        print("task num=%d" % len(ids))

        return dataSet

def clusterVec(taskdata,docX):
    X_techs=taskdata.techs
    X_lans=taskdata.lans

    print(X_techs.shape,docX.shape)
    X=np.concatenate((docX,X_techs),axis=1)
    X=np.concatenate((X,X_lans),axis=1)

    return X

def taskVec(taskdata,docX):
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

#save data and load data
def saveTaskVecData(X,taskids,dataname,choice=1):
    data={}
    data["size"]=len(taskids)
    data["taskids"]=taskids
    data["tasks"]=X
    with open("../data/clusterResult/"+dataname+"-taskData-"+str(choice)+".data","wb") as f:
        pickle.dump(data,f)


def genResultOfTasktype(tasktype,taskdata,choice):
    taskdata=TaskDataContainer(taskdata)
    taskdata.encodingFeature(choice)
    docX=taskdata.docs
    #kernel pca method
    print(tasktype,docX.shape,"before")
    kpca=decomposition.KernelPCA()
    docX=kpca.fit_transform(docX)
    print(tasktype,docX.shape,"after")
    print(tasktype,kpca.dual_coef_,kpca.n_components)

    #save vec representaion of all the tasks
    X=taskVec(taskdata,docX)
    saveTaskVecData(X,taskdata.ids,tasktype,choice)

    #cluster task based on its feature
    X=clusterVec(taskdata,docX)

    print("training for clustering, tasks size=%d"%len(X))
    clusterEXP=500
    IDClusters={}
    model=ClusteringModel()
    model.name=tasktype+"-clusteringModel-"+str(choice)
    n_clusters=max(1,len(taskdata.ids)//clusterEXP)

    for k in model.dataSet.keys():
        n_clusters=max(1,len(model.dataSet[k])//clusterEXP)
        localids=model.dataSet[k]
        localX=[]
        localIDs=[]
        for i in range(len(taskid)):
            id=taskid[i]
            if id in localids:
                localX.append(X[i])
                localIDs.append(id)
        localX=np.array(localX)
        model.trainCluster(X=localX,tasktype=k,n_clusters=n_clusters,minibatch=False)

        result=model.predictCluster(localX,k)
        for j in range(len(result)):
            r=result[j]
            t=k+str(r)
            if t not in IDClusters.keys():
                IDClusters[t]=[localIDs[j]]
            else:
                IDClusters[t].append(localIDs[j])

    model.saveModel()
    model.loadModel()
    # saving result
    print("saving clustering result")
    with open("../data/clusterResult/clusters" + str(choice) + ".data", "wb") as f:
        pickle.dump(IDClusters, f)

    with open("../data/clusterResult/clusters" + str(choice) + ".data", "rb") as f:
        taskidClusters=pickle.load(f)

    #plot result
    hist = [(k,len(taskidClusters[k])) for k in taskidClusters.keys()]
    for i in hist:
        print(i)
    plt.plot(np.arange(len(hist)),hist, marker='o')
    plt.title("choice=%d" % choice)
    plt.show()

def genResults():
    dataSet=initDataSet()
    typeinfo=list(dataSet.keys())

    choice=eval(input("1:LDA; 2:LSA \t"))

    for t in typeinfo:
        taskdata=TaskDataContainer(dataSet[t])
        multiprocessing.Process(target=genResultOfTasktype,args=(t,taskdata,choice)).start()

        #taskdata.encodingFeature(choice)



if __name__ == '__main__':
    genResults()

