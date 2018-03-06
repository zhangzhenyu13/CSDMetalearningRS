'''
this algorithm is aimed at sorting the list obeject under a given constraint
'''
import random
def compare_list(d1,d2):
    if d1[len(d1)-1]>d2[len(d2)-1]:
        return 1
    elif d1[len(d1)-1]<d2[len(d2)-1]:
        return -1
    else:
        return 0
def insertSort(A):
    #this is sub program for sorting of list Object in a user defined way
    for inserPos in range(0,len(A)):
        max=A[inserPos]
        maxPos=inserPos
        #print(A)
        for findPos in range(inserPos,len(A)):
            if compare_list(A[findPos],max)>0:
                maxPos=findPos
                max=A[maxPos]
        inserA=A[inserPos]
        A[inserPos]=A[maxPos]
        A[maxPos]=inserA
        #print(maxPos)
        inserPos=inserPos+1
    return A
#mergeSort
def merge(left,right):
    A=[]
    j=0
    i=0
    while i<len(left) and j<len(right):
        if compare_list(left[i],right[j])>=0: # left[i]>=right[j]:
            A.append(left[i])
            i=i+1
        else:
            A.append(right[j])
            j=j+1
    if len(left)>i:
        A=A+left[i:]
    if len(right)>j:
        A=A+right[j:]
    return A
def mergeSort(A):
    B=[]
    if len(A)>1:
        mid=int(len(A) / 2)
        left = A[0:mid]
        right = A[mid:]
        #print("L",left)
        #print("R",right)
        left=mergeSort(left)
        right=mergeSort(right)
        B=merge(left,right)
        return B
    else:
        return A
def mergeSort2(A):
    if len(A)<2:
        return A

    seg=1
    while seg<len(A):
        i=0
        while i<len(A)-seg:
            left=A[i:i+seg]
            right=A[i+seg:i+2*seg]
            #print("L",left)
            #print("R",right)
            A[i:i+2*seg]=merge(left,right)
            i=i+2*seg
        seg=seg*2

    return A
#quickSort
def sameCheck(A):
    #when all the elements in A are same, return True
    #else return False
    if len(A)==0:
        return True
    b=A[0]
    for a in A:
        if a!=b:
            return False
    return True
def quickSort(A):
    if len(A)<2:
        return A
    left=[]
    right=[]
    pivot=random.randint(0,len(A)-1)
    for a in A:
        if a<=A[pivot]:
            left.append(a)
        else:
            right.append(a)
    #print(pivot,A[pivot],A)
    #print("L",left)
    #print("R",right)
    if sameCheck(left):
        pass
    else:
        left=quickSort(left)
    if sameCheck(right):
        pass
    else:
        right=quickSort(right)

    return left+right
def quickSort2(A):
    if len(A)<2:
        return A
    tag=False
    slices=[A]
    while tag!=True:
        tag=True
        for i in range(len(slices)):
            slice=slices[i]
            if len(slice)>1 and sameCheck(slice)==False:
                tag=False
                pivot=random.randint(0,len(slice)-1)
                left=[]
                right=[]
                for a in slice:
                    if a<=slice[pivot]:
                        left.append(a)

                    else:
                        right.append(a)
                slices[i]=left
                slices.insert(i+1,right)
    B=[]
    for slice in slices:
        B=B+slice
    return B




def MapPos(value,bucketCount):
    # map a spcific record to a bucket
    pass

def bucket_sort(inputList):
    #this part mainly focus on assigning buckets and combining buckets
    bucketCount=int(len(inputList)/2)
    buckets = [[] for i in range(bucketCount)]
    outputList = []

    for e in inputList:
        buckets[MapPos(e,bucketCount)].append(e)

    for i in range(bucketCount):
        print(buckets[i])
        insertSort(buckets[i])

    for i in range(len(buckets)):
        outputList=outputList+buckets[i]

    return outputList

#for testing purpose
def main():
    A=[13,8,9,10,7,1,100,89,9,7]
    #A = [[1, 2], [2, 8], [7, 2], [6, 3],[5,89]]
    #print(insertSort(A))

    #B=bucket_sort(A)
    #print(B)

    #print(mergeSort2(A))
    print(quickSort2(A))

if __name__=="__main__":
    main()
