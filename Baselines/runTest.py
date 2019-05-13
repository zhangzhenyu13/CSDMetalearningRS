import warnings

warnings.filterwarnings("ignore")

from .CBC import buildCBC
from .CrowdRex import buildCrowdRex
from .DCW_DS import buildDCW_DS

def loadModelForTask(tasktype):
    dcwds=buildDCW_DS(tasktype)
    crowdrex=buildCrowdRex(tasktype)
    cbc=buildCBC(tasktype)
    return dcwds,crowdrex,cbc

if __name__ == '__main__':
    from ML_Models.ModelTuning import loadData,showMetrics,topKmetrics
    from ML_Models.UserMetrics import TopKMetrics
    from Utility import SelectedTaskTypes
    tasktypes=SelectedTaskTypes.loadTaskTypes()
    for tasktype in tasktypes["keeped"]:

        if "Code" in tasktype or "Assembly" in tasktype or "First2Finish" in tasktype:
            continue
        dcwds,crowdrex,cbc=loadModelForTask(tasktype)

        data=loadData(tasktype,2)

        mymetric=TopKMetrics(data.tasktype)

        #measuer model dcwds
        Y_predict2=dcwds.predict(data.testX)
        showMetrics(Y_predict2,data,0.5)
        topKmetrics(mymetric,Y_predict2,data)

        #measuer model dcwds
        Y_predict2=crowdrex.predict(data.testX)
        showMetrics(Y_predict2,data,0.5)
        topKmetrics(mymetric,Y_predict2,data)

        #measuer model dcwds
        Y_predict2=cbc.predict(data.testX)
        showMetrics(Y_predict2,data,0.5)
        topKmetrics(mymetric,Y_predict2,data)

