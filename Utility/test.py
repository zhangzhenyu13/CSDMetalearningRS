import json
import os
import numpy as np
import gc
if __name__ == '__main__':

    files=os.listdir("../data/UserGraph/fullGraph/")
    for file in files:
        with open("../data/UserGraph/fullGraph/"+file,"r") as f:
            data=None
            X=None
            gc.collect()
            data=json.load(f)

            #print(file,data["size"],data["users"])
            X=np.array(data["data"])
            n=np.sum(X!=0)
            p=np.sum(X>0)
            print(file,data["n_users"])
            print(p,n,n/X.size,np.min(X),np.max(X),np.mean(X[np.where(X!=0)]),np.mean(X[np.where(X>0)]),np.mean(X[np.where(X<0)]))

