import _pickle as pickle
import numpy as np

class Maskdata:
    #dig
    maskDig=[]
    #challenge
    maskLanguage=[]
    maskTech=[]
    maskTitle=[]
    maskDesc=[]
    maskDate=[]
    maskPrize=[]
    maskDiff=[]
    #user
    maskSkill=[]
    maskAge=[]
    maskMatchLan=[]
    maskMatchTech=[]
    maskReg=[]
    maskSub=[]
    maskWin=[]
    maskPerform=[]
    maskRegRank=[]
    maskSubRank=[]
    maskWinRank=[]

    @staticmethod
    def initMasks(dim):
        #construct vector mask
        #dig
        Maskdata.maskDig=np.ones(shape=(1,dim),dtype=np.bool)
        #challenge
        Maskdata.maskLanguage=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskTech=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskTitle=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskDesc=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskDate=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskPrize=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskDiff=np.ones(shape=(1,dim),dtype=np.bool)
        #user
        Maskdata.maskSkill=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskAge=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskMatchLan=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskMatchTech=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskReg=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskSub=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskWin=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskPerform=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskRegRank=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskSubRank=np.ones(shape=(1,dim),dtype=np.bool)
        Maskdata.maskWinRank=np.ones(shape=(1,dim),dtype=np.bool)

        #setting mask index
        masks=[
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
        with open("../data/masks","r") as f:
            lines=f.readlines()
            i=0
            for s in lines:
                if not s or not s.strip():
                    break
                if ":" in s.strip():
                    s=s.strip().split(":")
                    masks[i]=[ind for ind in range(int(s[0]),int(s[1]))]
                else:
                    s=s.strip().split()
                    masks[i]=list(map(lambda x:int(x),s))
                i+=1

        Maskdata.maskDig[:,masks[0]]=False
        #challenge
        Maskdata.maskLanguage[:,masks[0]]=False
        Maskdata.maskTech[:,masks[1]]=False
        Maskdata.maskTitle[:,masks[2]]=False
        Maskdata.maskDesc[:,masks[3]]=False
        Maskdata.maskDate[:,masks[4]]=False
        Maskdata.maskPrize[:,masks[5]]=False
        Maskdata.maskDiff[:,masks[6]]=False
        #user
        Maskdata.maskSkill[:,masks[7]]=False
        Maskdata.maskAge[:,masks[8]]=False
        Maskdata.maskMatchLan[:,masks[9]]=False
        Maskdata.maskMatchTech[:,masks[10]]=False
        Maskdata.maskReg[:,masks[11]]=False
        Maskdata.maskSub[:,masks[12]]=False
        Maskdata.maskWin[:,masks[13]]=False
        Maskdata.maskPerform[:,masks[14]]=False
        Maskdata.maskRegRank[:,masks[15]]=False
        Maskdata.maskSubRank[:,masks[16]]=False
        Maskdata.maskWinRank[:,masks[17]]=False

class ML_model:
    def __init__(self):
        self.model=None
        self.name=""
        self.threshold=0.5
        self.verbose=1
        self.mask=None

    def predict(self,X):
        '''
        predict the result based on given X
        :param X: input samples,(n,D)
        :return: given result, class or a real num
        '''

    def maskX(self,X):
        '''

        :param X: input samples, (n,D)
        :return: the masked index(False) using mask for each sample are set 0
        '''
        masked=np.array(self.mask*len(X),dtype=np.float).reshape(X.shape)
        X[masked==False]=0
        return X

    def trainModel(self,dataSet):
        pass

    def findPath(self):
        modelpath="../data/saved_ML_models/classifiers/"+self.name+".pkl"
        return modelpath
    def loadModel(self):
        with open(self.findPath(),"rb") as f:
            data=pickle.load(f)
            self.model=data["model"]
            self.name=data["name"]
    def saveModel(self):
        with open(self.findPath(),"wb") as f:
            data={}
            data["model"]=self.model
            data["name"]=self.name
            pickle.dump(data,f,True)

