import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import Utility.personalizedSort as ps

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
        #print()
        #print(X[:3])
        names=X[:,1]
        X=np.array(np.delete(X,1,1),dtype=np.float32)

        return X,names

    def setMatrixData(self,data):

        self.usermatrix=np.array(data["data"])
        self.usernames=np.array(data["users"])
        minmax=MinMaxScaler(feature_range=(0,1))
        self.usermatrix=minmax.fit_transform(self.usermatrix)

def rankOnDIG(data):
    '''
    :param data:{"data":i_matrix,"users":users_list}
    :return: rank
    '''

    poweruser=SelectPowerfulUser()
    poweruser.setMatrixData(data=data)
    X, names = poweruser.userRank()

    return X,names

