import pickle
def genFilterTypes():
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

    with open("../data/Statistics/typefilters.data","wb") as f:
        pickle.dump(filteredtypes,f)


def loadFilteredTypes():
    with open("../data/Statistics/typefilters.data","rb") as f:
        filters=pickle.load(f)

    return filters

if __name__ == '__main__':
    genFilterTypes()
    filters=loadFilteredTypes()
    print(len(filters),filters)


