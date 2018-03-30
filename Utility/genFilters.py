import pickle
def genFilterTypes():
    filteredtypes=[
        'Studio Other',
        'Front-End Flash',
        'Spec Review',
        'Legacy',
        'Copilot Posting',
        'Marathon Match',
        'Design First2Finish'
    ]
    with open("../data/Statistics/filteredChallengeTypes.data","wb") as f:
        pickle.dump(filteredtypes,f)
def loadFilteredTypes():
    with open("../data/Statistics/filteredChallengeTypes.data","rb") as f:
        filters=pickle.load(f)
    return filters

if __name__ == '__main__':
    genFilterTypes()
    print(loadFilteredTypes())
