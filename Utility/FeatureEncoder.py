from Utility.personalizedSort import MySort
import numpy as np
def onehotFeatures(data):
    '''
    :param data:str data
    :return: one-hot feature_dict
    '''
    c = {}
    for r in data:
        if r is None or r=="":
            continue
        xs = r.split(",")
        for x in xs:
            if x in c.keys():
                c[x] += 1
            else:
                c[x] = 1
    #print(data)
    #print("doc item",c)
    rmK=[]
    for k in c.keys():
        if "Other" in k:
            rmK.append(k)
    for k in rmK:
        del c[k]

    features=[]
    for k in c.keys():
        features.append([k,c[k]])
    ms=MySort(features)
    ms.compare_vec_index=-1
    features=ms.mergeSort()
    features_dict={}

    for i in range(len(features)):
        item=features[i]
        features_dict[item[0]]=item[1]
    print(features_dict)

    features_dict={}
    for i in range(len(features)):
        features_dict[features[i][0]]=i

    return features_dict

def EncodeByDictOne(record,feature_dict,clip_num):
    x=np.zeros(shape=min(len(feature_dict),clip_num))
    if record is None:
        record=""
    record=record.split(",")
    for r in record:
        if r in feature_dict.keys():
            index=feature_dict[r]
            if index<len(x):
                x[index]=1
    return x

def EncodeByDict(data,feature_dict,clip_num):
    X=[]

    for i in range(len(data)):
        record=data[i]
        x=EncodeByDictOne(record,feature_dict,clip_num)
        X.insert(i,x)

    X=np.array(X)

    return X
