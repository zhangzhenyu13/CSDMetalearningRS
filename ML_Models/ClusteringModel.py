# Do the actual clustering
from ML_Models.Model_def import ML_model
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
import numpy as np

class ClusteringModel(ML_model):
    def __init__(self):
        ML_model.__init__(self)

    def trainCluster(self,X,n_clusters,minibatch=False):

        if minibatch:
            km = MiniBatchKMeans(n_clusters=n_clusters, verbose=False)
        else:
            km = KMeans(n_clusters=n_clusters,verbose=False)

        #print("Clustering sparse data with %s" % km)
        km.fit(X)
        print()
        self.model=km
    def predictCluster(self,X):
        print(self.name+" is clustering %d tasks"%len(X))
        return  self.model.predict(X)

    def findPath(self):
        modelPath="../data/saved_ML_models/clusterModels/"+self.name+".pkl"
        return modelPath
