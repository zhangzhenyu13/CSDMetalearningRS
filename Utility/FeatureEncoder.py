from scipy import sparse
from Utility.personalizedSort import *

def onehotFeatures(data,feature_num=20):
    '''
    :param data:str data
    :return: one-hot vector representation
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

    while len(c)>feature_num:
        minNum=1e+5
        rmK=None
        for k in c.keys():
            if c[k]<minNum :
                rmK=k
                minNum=c[k]

        del c[rmK]

    c=c.keys()
    i_c = {}
    count = 0
    for i in c:
        i_c[i] = count
        count += 1
    #print(i_c)
    X = sparse.dok_matrix((len(data), feature_num))
    row = 0
    for r in data:

        if r is None:
            row += 1
            continue

        xs = r.split(",")
        for x in xs:
            if x not in c:
                continue
            col = i_c[x]
            try:
                X[row, col] = 1
            except:

                print("~~~~~~",row,col,X.shape)
        row += 1
    #print("one-hot feature size=%d"%(len(c)),"removed feature size=%d"%(len(rmKs)))

    return X.toarray()


