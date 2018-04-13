from DataPrepare.ConnectDB import *
from Utility.FeatureEncoder import onehotFeatures
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing,_pickle as pickle
from DataPrepare.DataContainer import TaskDataContainer
from ML_Models.ClusteringModel import ClusteringModel
from ML_Models.DocTopicsModel import LDAFlow
from Utility.TagsDef import *
from Utility import SelectedTaskTypes

warnings.filterwarnings("ignore")

def showData(X):
    import Utility.personalizedSort as ps
    m_s=ps.MySort(X)
    m_s.compare_vec_index=-1
    y=m_s.mergeSort()
    x=np.arange(len(y))
    plt.plot(x,np.array(y)[:,0])
    plt.show()

def genGlobalFeatures(techs,lans,docs,ids):
    print("saving global encoding")
    techs_enc=onehotFeatures(techs)
    lans_enc=onehotFeatures(lans)
    print(techs_enc)
    print(lans_enc)
    lda=LDAFlow()
    lda.name="global"
    lda.train_doctopics(docs)
    with open("../data/TaskInstances/GlobalEncoding.data","wb") as f:
        pickle.dump({"techs":techs_enc,"lans":lans_enc,"ids":ids},f,True)

def initDataSet(return_all=False,genFeatures=False):
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

            if data[9].replace("/","_") in SelectedTaskTypes.filteredtypes:
                continue
            if data[1] is None:
                continue
            ids.append(data[0])
            docs.append(data[2]+" "+data[1])

            if data[3]>50:
                durations.append(50)
            elif data[3]<1:
                durations.append(1)
            else:
                durations.append(data[3])

            techs.append(data[4])
            lans.append(data[5])

            if data[6]!='':
                prize=np.sum(eval(data[6]))
                if prize>6000:
                    prize=6000
                if prize<1:
                    prize=1
                prizes.append(prize)
            else:
                prizes.append(1.)

            if data[7]<1:
                startdates.append(1)
            else:
                startdates.append(data[7])

            if data[8]>0.6:
                diffdegs.append(0.6)
            else:
                diffdegs.append(data[8])

            tasktypes.append(data[9].replace("/","_"))

        print("task size=",len(ids),len(docs),len(techs),len(lans),len(startdates),len(durations),len(prizes),len(diffdegs),len(tasktypes))


        if genFeatures:
            genGlobalFeatures(techs,lans,docs,ids)
        if return_all:
            container=TaskDataContainer(typename='global')

            container.ids=ids
            container.docs=docs
            container.techs=techs
            container.lans=lans
            container.startdates=startdates
            container.durations=durations
            container.prizes=prizes
            container.diffdegs=diffdegs
            return container
        #adding to corresponding type
        print("init data set types")

        dataSet={}
        for i in range(len(tasktypes)):
            t=tasktypes[i].replace("/","_")
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

        print("slected types(%d):"%len(dataSet),dataSet.keys())
        print()

        return dataSet

def clusterVec(taskdata,docX):

    X_techs=taskdata.techs_vec
    X_lans=taskdata.lans_vec
    X_diffdegs=np.reshape(np.log(taskdata.diffdegs),newshape=(len(docX),1))
    X_durations=np.reshape(np.log(taskdata.durations),newshape=(len(docX),1))
    X_prizes=np.reshape(np.log(taskdata.prizes),newshape=(len(docX),1))
    X_startdates=np.reshape(np.log(taskdata.startdates),newshape=(len(docX),1))
    print("cluster shape: docs,techs,lans",docX.shape,X_techs.shape,X_lans.shape)
    X=np.concatenate((docX,X_techs),axis=1)
    X=np.concatenate((X,X_lans),axis=1)
    X=np.concatenate((X,X_diffdegs),axis=1)
    X=np.concatenate((X,X_durations),axis=1)
    X=np.concatenate((X,X_prizes),axis=1)
    X=np.concatenate((X,X_startdates),axis=1)
    data={}
    data["ids"]=taskdata.ids
    data["X"]=X

    with open("../data/TaskInstances/taskContent/"+taskdata.taskType+"-taskData.data","wb") as f:
        pickle.dump(data,f,True)

    return X


#save data content as a vector
def saveTaskData(taskdata):
    data={}
    docX=np.array(taskdata.docs)

    data["docX"]=docX

    lans=[None for i in range(len(taskdata.ids))]
    techs=[None for i in range(len(taskdata.ids))]
    for i in range(len(taskdata.ids)):
        lans[i]=taskdata.lans[i].split(",")
        techs[i]=taskdata.techs[i].split(",")

    data["lans"]=lans
    data["techs"]=techs
    data["diffdegs"]=taskdata.diffdegs
    data["startdates"]=taskdata.startdates
    data["durations"]=taskdata.durations
    data["prizes"]=taskdata.prizes
    data["ids"]=taskdata.ids

    with open("../data/TaskInstances/taskDataSet/"+taskdata.taskType+"-taskData.data","wb") as f:
        pickle.dump(data,f,True)

def genResultOfTasktype(tasktype,taskdata,choice):

    taskdata.encodingFeature(choice)
    docX=taskdata.docs
    saveTaskData(taskdata)

    clusterEXP=800
    n_clusters=max(1,len(taskdata.ids)//clusterEXP)
    if n_clusters==1:
        return

    #cluster task based on its feature
    X=clusterVec(taskdata,docX)

    model=ClusteringModel()
    model.name=tasktype+"-clusteringModel"

    model.trainCluster(X=X,n_clusters=n_clusters,minibatch=False)

    model.saveModel()
    model.loadModel()
    result=model.predictCluster(X)

    IDClusters={}
    ClustersData={}

    for i in range(len(result)):
        t=result[i]

        if t not in IDClusters.keys():
            IDClusters[t]=[taskdata.ids[i]]

            container=TaskDataContainer(typename=t)
            container.ids.append(taskdata.ids[i])
            container.docs.append(taskdata.docs[i])
            container.techs.append(taskdata.techs[i])
            container.lans.append(taskdata.lans[i])
            container.startdates.append(taskdata.startdates[i])
            container.durations.append(taskdata.durations[i])
            container.prizes.append(taskdata.prizes[i])
            container.diffdegs.append(taskdata.diffdegs[i])

            ClustersData[t]=container
        else:
            IDClusters[t].append(taskdata.ids[i])

            container=ClustersData[t]
            container.ids.append(taskdata.ids[i])
            container.docs.append(taskdata.docs[i])
            container.techs.append(taskdata.techs[i])
            container.lans.append(taskdata.lans[i])
            container.startdates.append(taskdata.startdates[i])
            container.durations.append(taskdata.durations[i])
            container.prizes.append(taskdata.prizes[i])
            container.diffdegs.append(taskdata.diffdegs[i])

            ClustersData[t]=container

    # saving result
    print("saving clustering result")
    for t in ClustersData.keys():
        container=ClustersData[t]
        container.taskType=tasktype+"#"+str(t)
        saveTaskData(container)

    print("saving cluster plot result")
    plt.figure(tasktype)
    y=[]
    for i in range(n_clusters):
        y.append(len(IDClusters[i]))
        print(tasktype,"#%d"%i,"size=%d"%len(IDClusters[i]))

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

    choice=eval(input("1:LDA; 2:LSA \t"))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    tasktypes=dataSet.keys()
    for t in tasktypes:
        taskdata=dataSet[t]
        tasktype=taskdata.taskType

        #print(taskdata.ids);exit(10)
        multiprocessing.Process(target=genResultOfTasktype,args=(tasktype,taskdata,choice)).start()

        #genResultOfTasktype(tasktype=tasktype,taskdata=taskdata,choice=choice)

if __name__ == '__main__':
    #genResults();exit(10)

    dataSet=initDataSet(True)
    dataSet.encodingFeature(1)
    saveTaskData(dataSet)
