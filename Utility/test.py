import json
import os
import numpy as np
import gc
from numpy import linalg as LA
if __name__ == '__main__':

    files=os.listdir("../data/UserGraph/initGraph/")
    for file in files:
        with open("../data/UserGraph/initGraph/"+file,"r") as f:
            data=None
            X=None
            gc.collect()
            data=json.load(f)

            matrix=data["data"]
            w=LA.eigvals(matrix)
            #print(w)
            maxeig=np.max(w)
            print(file,"max eig=",maxeig,np.real(maxeig))

