# Do the actual clustering
from ML_Models.Model_def import ML_model
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
import numpy as np

class ClusteringModel(ML_model):
    def __init__(self,tasktype):
        ML_model.__init__(self)
        self.model={}


    def trainCluster(self,X,n_clusters=1,minibatch=False):

        if minibatch:
            km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=False)
        else:
            km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                        verbose=False)

        print("Clustering sparse data with %s" % km)
        km.fit(X)
        print()
        self.model=km
    def predictCluster(self,X,tasktype):
        return  self.model[tasktype].predict(X)

    def findPath(self):
        modelPath="../data/saved_ML_models/clusterModels/"+self.name+".pkl"
        return modelPath
