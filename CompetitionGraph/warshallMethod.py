import math
import numpy as np
import json

def loadGraph():
    f = open("../data/warshalltest.json")
    graphFile=json.load(f)
    size = graphFile["size"]
    graphMatrix = graphFile["data"]

    f.close()

    resultMatrix = np.array(graphMatrix)
    return resultMatrix


def warshallRun(resultMatrix):
    size = len(resultMatrix)
    PL = np.zeros(shape=(size, size))
    for k in range(size):
        for i in range(size):
            for j in range(size):
                curp = resultMatrix[i][k] * math.exp(PL[i][k]) * resultMatrix[k][j]
                if math.fabs(resultMatrix[i][j]) < math.fabs(curp):
                    resultMatrix[i][j] = curp
                    PL[i][j] = k

    resultFile = open("../data/warshallresult.json", 'w')
    result={}
    result["size"]=size
    result["data"]=resultMatrix.tolist()
    json.dump(result,resultFile,ensure_ascii=False)

    resultFile.close()


if __name__ == '__main__':
    result_m = loadGraph()
    warshallRun(result_m)

