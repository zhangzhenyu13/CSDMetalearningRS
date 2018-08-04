import pickle
import numpy as np
import time,random,copy
metapath="../data/MetaData/"


def getAllDatasets():
    import os
    files=os.listdir(metapath)
    datasets=[]
    for file in files:
        datasets.append(file.replace(".pkl",""))

    return datasets

def loadMetaData(datasetname):
    file=metapath+datasetname+".pkl"
    with open(file,"rb") as f:
        metadata=pickle.load(f)
    metadata=np.array(metadata,dtype=np.float32)
    return metadata
def getBestPerformance(metadata):
    bestacc3=0
    bestacc5=0
    bestacc10=0
    bestmrr=0

    for md in metadata:
        if md[-1]>bestmrr:
            bestmrr=md[-1]
        if md[-2]>bestacc10:
            bestacc10=md[-2]
        if md[-3]>bestacc5:
            bestacc5=md[-3]
        if md[-4]>bestacc3:
            bestacc3=md[-4]
    return bestacc3,bestacc5,bestacc10,bestmrr

def getPerformance(features,metadata):
    featurelen=len(features)
    for md in metadata:
        for i in range(featurelen):
            if md[i]==features[i]:
                return (md[-4],md[-3],md[-2],md[-1])
    return None


#search related components

class PSampler:
    def __init__(self,samples=None):
        '''

        :param samples: dict represent the importance of each sample, e.g. {0:0.6,1:0.8}
        '''
        if samples is None:
            samples={0:1,1:1,2:1,3:1,4:1}
        self.size=100
        self.samples=[]
        for k in samples:
            count=int(samples[k]*self.size)
            for j in range(count):
                self.samples.append(k)

        random.shuffle(self.samples)

    def getIndicator(self):
        index=random.randint(0,len(self.samples)-1)
        return self.samples[index]

    def praiseIndicator(self,indicator,k):
        for i in range(k):
            self.samples.append(indicator)
        random.shuffle(self.samples)

    def penalizeIndicator(self,indicator,k):
        self.samples.sort()

        left,right=-1,-1
        for i in range(len(self.samples)):
            if self.samples[i]==indicator:
                left=i
                break

        if left==-1:
            return

        for i in range(left+1,len(self.samples)):
            if self.samples[i]!=indicator:
                right=i
                break

        if right==-1:
            right=len(self.samples)

        if right-left>=k:
            self.samples=self.samples[:left]+self.samples[right:]
        self.samples=self.samples[:left]+self.samples[left+k:]
        random.shuffle(self.samples)

class FeatureNode:
    metric=-1# -4,-3,-2,-1

    def __init__(self):
        self.feature=[1,1,1,0.1,0.1]
        self.perf=0
        self.stepper=PSampler()
        self.badCount=0
        self.history=[]
        self.prevNode=None
        self.IterNum=1
        self.f_sel=0
        self.fetchPerformance()

    def fetchPerformance(self):
        perf=getPerformance(self.feature,metadata)
        self.perf=perf[self.metric]

    def checkSame(self,node):
        for i in range(len(self.feature)):
            if node.feature[i]!=self.feature[i]:
                return False
        return True
    def getNextStep(self,increase=True):
        f_sel=self.stepper.getIndicator()
        node=copy.deepcopy(self)
        if f_sel in (0,1,2):
            if node.feature[f_sel]>=2:
                node.feature[f_sel]-=1
            elif node.feature[f_sel]<=0:
                node.feature[f_sel]+=1
            else:
                if increase:
                    node.feature[f_sel]-=1
                else:
                    node.feature[f_sel]+=1
        else:
            if node.feature[f_sel]>=1:
                node.feature[f_sel]-=0.1
            elif node.feature[f_sel]<=0:
                node.feature[f_sel]+=0.1
            else:
                if increase==0:
                    node.feature[f_sel]-=0.1
                else:
                    node.feature[f_sel]+=0.1
        node.history.append(self)
        node.prevNode=self
        node.IterNum=self.IterNum+1

        self.f_sel=f_sel

        return node

    def SearchNext(self):
        node=None
        visited=True
        maxIterNum=100
        iterCount=0
        while visited and iterCount<maxIterNum:
            print("Inner IterNum=",iterCount)
            node=self.getNextStep()
            visited=False
            for hn in self.history:
                if node.checkSame(hn):
                    visited=True
                    break
            iterCount+=1

        return node


def gradientSearch(metadata):
    maxBad=2
    maxIter=100
    prize=2
    fine=3
    t=1
    bestperf=0
    node=FeatureNode()
    while t<=maxIter:
        print("Outer IterNum=",t,"best perf=",bestperf)
        cur=node.SearchNext()
        cur.fetchPerformance()

        if cur.perf>node.perf:
            cur.stepper.praiseIndicator(cur.f_sel,prize)
        elif cur.perf<node.perf:
            cur.stepper.penalizeIndicator(cur.f_sel,fine)
            cur.badCount+=1
        else:
            cur.stepper.penalizeIndicator(cur.f_sel,fine-1)

        if cur.badCount>=maxBad:
            print("too much bad, prune the branch")
            hn=cur.history
            cur=cur.prevNode
            cur=cur.prevNode
            cur.history=hn

        if cur.perf>bestperf:
            bestperf=cur.perf
        node=cur
        t+=1

    print("random learner search best perf",bestperf)

    pass


if __name__ == '__main__':
    ds=getAllDatasets()
    ds.sort()
    print(ds)
    for d in ds:
        print(d)
        metadata=loadMetaData(d)
        print(metadata.shape)
        bestperf=getBestPerformance(metadata)
        print("global",bestperf)

        #gradientSearch(metadata)
        print()



