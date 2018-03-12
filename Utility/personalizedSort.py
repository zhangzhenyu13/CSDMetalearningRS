'''
this algorithm is aimed at sorting the list obeject under a given constraint
'''
import numpy as np
import random
class MySort:
    def __init__(self,A):
        self.compare_vec_index = -1
        self.A=A
    def compare_list(self,d1,d2):
        index=self.compare_vec_index
        if d1[index]>d2[index]:
            return 1
        elif d1[index]<d2[index]:
            return -1
        else:
            return 0
    def insertSort(self):
        #this is sub program for sorting of list Object in a user defined way
        A=self.A
        for inserPos in range(0,len(A)):
            max=A[inserPos]
            maxPos=inserPos
            #print(A)
            for findPos in range(inserPos,len(A)):
                if self.compare_list(A[findPos],max)>0:
                    maxPos=findPos
                    max=A[maxPos]
            inserA=A[inserPos]
            A[inserPos]=A[maxPos]
            A[maxPos]=inserA
            #print(maxPos)
            inserPos=inserPos+1
        return A
    #mergeSort
    def merge(self,left,right):
        A=[]
        j=0
        i=0
        while i<len(left) and j<len(right):
            if self.compare_list(left[i],right[j])>=0: # left[i]>=right[j]:
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
    def mergeSort(self):
        A=self.A
        if len(A)<2:
            return A

        seg=1
        while seg<len(A):
            i=0
            while i<len(A)-seg:
                left=A[i:i+seg]
                right=A[i+seg:i+2*seg]

                A[i:i+2*seg]=self.merge(left,right)
                i=i+2*seg
            seg=seg*2

        return A
    #quickSort
    def sameCheck(self,A):

        #when all the elements in A are same, return True
        #else return False
        if len(A)==0:
            return True
        b=A[0]
        for a in A:
            if a!=b:
                return False
        return True

    def quickSort(self):#correctness is not guaranteed
        A=self.A
        if len(A)<2:
            return A
        tag=False
        slices=[A]
        while tag!=True:
            tag=True
            for i in range(len(slices)):
                slice=slices[i]
                if len(slice)>1 and self.sameCheck(slice)==False:
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

#for testing purpose
def main():

    A = [[1, 2], [2, 8], [7, 2], [6, 3],[5,89]]

    mysort=MySort(A)
    mysort.compare_vec_index=-1
    A=mysort.mergeSort()
    A=np.array(A)
    print(A)
    print(A[:,-1])
    print(A[:,-2])

if __name__=="__main__":
    main()
