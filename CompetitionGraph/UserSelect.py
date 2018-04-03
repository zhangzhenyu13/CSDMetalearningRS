import networkx as nx
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import Utility.personalizedSort as ps
import matplotlib.pyplot as plt
import pickle

class SelectPowerfulUser:

    def __init__(self):
        self.usermatrix=None
        self.usernames=None

    def userRank(self):

        G = nx.DiGraph(self.usermatrix)
        pr = nx.pagerank_numpy(G, alpha=0.9)
        #print(pr)
        X=[]
        for i in range(len(pr)):
            #index,name,rank score
            X.insert(i,(i,self.usernames[i],pr[i]))

        m_s=ps.MySort(X)
        m_s.compare_vec_index=-1
        X=m_s.mergeSort()
        X=np.array(X)
        print()
        #print(X[:3])
        names=X[:,1]
        X=np.array(np.delete(X,1,1),dtype=np.float32)
        return X,names

    def setMatrixData(self,data):

        self.usermatrix=np.array(data["data"])
        self.usernames=np.array(data["users"])
        minmax=MinMaxScaler(feature_range=(0,1))
        self.usermatrix=minmax.fit_transform(self.usermatrix)

def rankLocalSelectTest(users,poweruser):

    with open("../data/UserGraph/initGraph/localGraph" + str(choice) + ".data", "rb") as f:
        data = pickle.load(f)
    selectedUsers = {}
    for k in data.keys():
        print(k,"size=",data[k]["size"])

        poweruser.setMatrixData(data=data[k])
        X, names = poweruser.userRank()
        x = np.arange(len(X))[:1000]
        y1 = []
        selectedUsers[k] = []
        for i in range(len(X)):
            r=X[i]
            name = names[i]
            subNum = data[k]["avgsubnum"][int(r[0])]
            # print(name,subNum)
            y1.append(subNum)

            selectedUsers[k].append(name)

        selectedUsers[k] = selectedUsers[k][:1000]

        picpath = "../data/pictures/UserRanks/"
        plt.xlabel("user rank")
        plt.ylabel("user submit num")
        y1 = y1[:1000]
        plt.title("submission disrtibution of top 1000")
        plt.plot(x, y1)
        # plt.show()
        plt.savefig(picpath + "sub_" + str(k) + ".jpg")
        plt.gcf().clear()


    with open("../data/Instances/localSelectedUsers.pkl", "wb") as f:
        pickle.dump(selectedUsers, f)

def rankGlobalSelect():
    with open("../data/UserGraph/initGraph/globalGraph" + str(choice) + ".data", "rb") as f:
        data = pickle.load(f)
    poweruser.setMatrixData(data=data)
    X, names = poweruser.userRank()
    x = np.arange(len(X))[:1000]
    y1 = []
    selectedUsers = []
    for i in range(len(X)):
        r = X[i]
        name = names[i]
        subNum = data["avgsubnum"][int(r[0])]
        # print(name,subNum)

        y1.append(subNum)

        selectedUsers.append(name)

    selectedUsers= selectedUsers[:1000]

    picpath = "../data/pictures/UserRanks/"
    plt.xlabel("user rank")
    plt.ylabel("user submit num")
    y1 = y1[:1000]
    plt.title("submission disrtibution of top 1000")
    plt.plot(x, y1)
    # plt.show()
    plt.savefig(picpath + "sub_Global" + ".jpg")
    plt.gcf().clear()

    with open("../data/Instances/globalSelectedUsers.pkl", "wb") as f:
        pickle.dump(selectedUsers, f)

if __name__ == '__main__':
    poweruser=SelectPowerfulUser()
    users=Users()
    choice = 1

    print(len(users.name[np.where(users.submissionNum>0)]))
    print(len(users.name[np.where(users.winNum>0)]))

    rankLocalSelectTest(users=users,poweruser=poweruser)


