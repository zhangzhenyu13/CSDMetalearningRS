import networkx as nx
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from DataPrepare.Entity import Users
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
        X=[]
        for i in range(len(pr)):
            X.insert(i,(i,self.usernames[i],pr[i]))

        m_s=ps.MySort(X)
        m_s.compare_vec_index=-1
        X=m_s.mergeSort()
        X=np.array(X)
        print()

        names=X[:,1]
        X=np.array(np.delete(X,1,1),dtype=np.float32)
        return X,names

    def loadMatrixData(self,choice,k):

        with open("../data/UserGraph/initGraph/initG_" + str(choice) + "_" + str(k) + ".json", "r") as f:
            data=json.load(f)
        print("size=",data["size"])
        self.usermatrix=np.array(data["data"])
        self.usernames=np.array(data["users"])
        for i in range(len(self.usermatrix)):
            Min=np.min(self.usermatrix[i])
            Max=np.max(self.usermatrix[i])
            if Min!=Max:
                self.usermatrix[i]=100*(self.usermatrix[i]-Min)/(Max-Min)

    def Sub_Win_RankSelect(self,users):
        pass

    def selectUserFeatures(self,names,choice,k_no):
        with open("../data/Instances/task_user" + str(choice) + ".data" + str(k_no), "rb") as f:
            data=pickle.load(f)
            user_vec=data["users"]
            username=data["names"]
            submits=data["submits"]
            ranks=data['ranks']
        featuredata=[]
        for i in range(len(names)):
            name=names[i]
            for j in range(1,len(username)+1):
                if name==username[-j]:
                    x=user_vec[-j][:8]+[submits[-j],ranks[-j]]
                    featuredata.append(x)
                    break
        return np.array(featuredata)

def rankSelectTest(users,poweruser):
    choice = 1
    n_clusters = 20
    selectedUsers = {}
    for k in range(n_clusters):
        poweruser.loadMatrixData(choice=choice, k=k)
        X, names = poweruser.userRank()
        x = np.arange(len(X))[:1000]
        y1 = []
        y2 = []
        selectedUsers[k] = []
        for r in X:

            name = names[int(r[0])]
            subNum = users.submissionNum[np.where(users.name == name)]
            winNum = users.winNum[np.where(users.name == name)]
            # print(name,subNum,winNum)
            if len(subNum) > 0:
                y1.append(subNum[0])
            else:
                y1.append(0)
            if len(winNum) > 0:
                y2.append(winNum[0])
            else:
                y2.append(0)
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
        plt.ylabel("user win num")
        y2 = y2[:1000]
        plt.title("winner disrtibution of top 1000")
        plt.plot(x, y2)
        # plt.show()
        plt.savefig(picpath + "win_" + str(k) + ".jpg")
        plt.gcf().clear()

    with open("../data/Instances/selectedUsersOnRank.pkl", "wb") as f:
        selectedUsers["n_clusters"] = n_clusters
        pickle.dump(selectedUsers, f)

if __name__ == '__main__':
    poweruser=SelectPowerfulUser()
    users=Users()

    print(len(users.name[np.where(users.submissionNum>0)]))
    print(len(users.name[np.where(users.winNum>0)]))

    rankSelectTest(users=users,poweruser=poweruser)


