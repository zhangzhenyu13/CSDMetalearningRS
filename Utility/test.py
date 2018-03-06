import json
import os
import numpy as np

if __name__ == '__main__':

    files=os.listdir("../data/UserGraph/initGraph/")
    for file in files:
        with open("../data/UserGraph/initGraph/"+file,"r") as f:
            data=json.load(f)
            #print(file,data["size"],data["users"])
            X=np.array(data["data"])
            n=np.sum(X!=0)
            p=np.sum(X>0)
            print(p,n,n/X.size,np.min(X))

