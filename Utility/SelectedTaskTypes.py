import _pickle as pickle,os
import numpy as np

filteredtypes=[
        'Web Design',
        'Banners_Icons',
        'Application Front-End Design',
        'Studio Other',
        'Logo Design',
        'Wireframes',
        'Print_Presentation',
        'Widget or Mobile Screen Design',
        'Front-End Flash',
        'Test Scenarios',
        'RIA Build Competition',
        'Specification',
        'Spec Review',
        'Idea Generation',
        'Legacy',
        'Copilot Posting',
        'Marathon Match',
        'Design First2Finish'
]

def genFinalTaskTypes():

    alltypes=np.array(os.listdir("../data/TaskInstances/taskDataSet/"))
    for i in range(len(alltypes)):
        t=alltypes[i]
        pos=t.find("-")
        alltypes[i]=t[:pos]

    finaltypes=[]
    clustertypes=[]

    for t in alltypes:
        if "#" not in t:
            finaltypes.append(t)
        else:
            pos=t.find("#")
            clustertypes.append(t)



    print(len(finaltypes),finaltypes)
    print(len(clustertypes),clustertypes)
    print("First2Finish" in finaltypes,"First2Finish" in clustertypes)
    #exit(10)
    with open("../data/Statistics/taskTypes.data","wb") as f:
        pickle.dump({"keeped":finaltypes,"clustered":clustertypes},f)



def loadTaskTypes():
    with open("../data/Statistics/taskTypes.data","rb") as f:
        tasktypes=pickle.load(f)
    return tasktypes

if __name__ == '__main__':

    genFinalTaskTypes()
    tasktypes=loadTaskTypes()
    print(tasktypes["clustered"])
    print(tasktypes["keeped"])
