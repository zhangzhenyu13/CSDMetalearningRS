from scipy import sparse

def onehotFeatures(data,threshold_num=5):
    '''
    :param data:str data
    :return: one-hot vector representation
    '''
    c = {}
    for r in data:
        if r is None:
            continue
        xs = r.split(",")
        for x in xs:
            if x in c.keys():
                c[x] += 1
            else:
                c[x] = 1

    rmKs=[]
    for k in c.keys():
        if c[k]<threshold_num :
            rmKs.append(k)
    for k in rmKs:
        del c[k]
    c=c.keys()
    i_c = {}
    count = 0
    for i in c:
        i_c[i] = count
        count += 1
    X = sparse.dok_matrix((len(data), count))
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
            X[row, col] = 1
        row += 1
    #print("one-hot feature size=%d"%(len(c)),"removed feature size=%d"%(len(rmKs)))
    return X.toarray()
