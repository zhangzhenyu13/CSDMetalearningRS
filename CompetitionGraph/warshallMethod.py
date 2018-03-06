import math
import numpy as np
import json
import os
import time
import multiprocessing
def loadGraph(gfile):
    f = open("../data/UserGraph/initGraph/"+gfile,"r")
    graphFile=json.load(f)
    size = graphFile["size"]
    graphMatrix = graphFile["data"]
    f.close()
    print("size=%d"%size)
    resultMatrix = np.array(graphMatrix)
    return resultMatrix


def warshallRun(resultMatrix,gfile):
    size = len(resultMatrix)
    PL = np.zeros(shape=(size, size))
    for k in range(size):
        for i in range(size):
            for j in range(size):
                curp = resultMatrix[i][k] * math.exp(PL[i][k]) * resultMatrix[k][j]
                if math.fabs(resultMatrix[i][j]) < math.fabs(curp):
                    resultMatrix[i][j] = curp
                    PL[i][j] = k

    resultFile = open("../data/UserGraph/fullGraph/full"+gfile[4:], 'w')
    result={}
    result["size"]=size
    result["data"]=resultMatrix.tolist()
    json.dump(result,resultFile,ensure_ascii=False)

    resultFile.close()


if __name__ == '__main__':
    gfiles=os.listdir("../data/UserGraph/initGraph/")

    for gfile in gfiles:
        t0=time.time()
        print("fill in",gfile)
        result_m = loadGraph(gfile)
        warshallRun(result_m,gfile)
        print("finished in %ds"%(time.time()-t0))

