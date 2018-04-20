# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np
from scipy.misc import factorial as fact
lines = np.loadtxt("HeartDiseaseData.txt")
XNominal = lines[:, [1, 2, 5, 6, 8, 10,11,12]]
XNumerical = lines[:, [0, 3, 4, 7, 9]]
mNominal=8
mNumerical=5
Y = lines[:, [13]]
m = 13
N = 270
number_of_clusters = 7
for i in xrange(0,N):
        Y[i]-=1

# lines = np.loadtxt("Balloon.txt")
# XNominal = lines[:, [0, 1, 2, 3]]
# XNumerical=[]
# Y = lines[:, [4]]
# m = 4
# mNominal=4
# mNumerical=0
# N = 16
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# a = np.array( [0,1,1,0,2,2,2,2,0,1] )
# b = np.array( [6,5.5,7.5,6,10.5,9,7.5,9, 7.5,7.5] )

def preprocessing1(a):
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    elementsAndFrequencies=np.column_stack((unique_elements,counts_elements))
    arr = elementsAndFrequencies[elementsAndFrequencies[:,1].argsort()]
    n=10;
    rows=np.shape(arr)[0];
    previousVal=0;
    j=0;
    temp=[];
    for i in xrange(0,rows):
        if(j>i):
            temp.append(previousVal)
            continue
        sameFrequencyCount=1;
        for j in xrange(i+1,rows):
            if(arr[j][1]==arr[i][1]):
                sameFrequencyCount=sameFrequencyCount+1
            else:
                break
        val=arr[i][1]* ((arr[i][1])-1)  ;
        val=val*1.0/(n*(n-1)*1.0);
        val=val* sameFrequencyCount
        val=val+previousVal
        temp.append(val)
        previousVal=val;

    add=arr[:,0]
    temp=np.column_stack((add,temp))
    return temp

def preprocessing2(b):
    n=10
    unique_elements, counts_elements = np.unique(b, return_counts=True)
    elementsAndFrequencies=np.column_stack((unique_elements,counts_elements))
    arr = elementsAndFrequencies[elementsAndFrequencies[:,0].argsort()]
    rows=np.shape(arr)[0];
    frequencyMatrix=np.zeros((rows,rows))
    for i in xrange(0,rows):
        for j in xrange(i,rows):
            if i==j:
                frequencyMatrix[i][j]=arr[i][1];
            else:
                frequencyMatrix[i][j]=frequencyMatrix[i][j-1]+arr[j][1];
    temp=np.zeros((rows,rows))
    for i in xrange(0,rows):
        for j in xrange(i,rows):
            for k in xrange(0,rows):
                for l in xrange(k,rows):
                    diff1=(arr[j][0]-arr[i][0])
                    diff2=(arr[l][0]-arr[k][0])
                    population1=arr[j][1]+arr[i][1]
                    if diff2<=diff1:
                        if diff2==0:
                            valueToAdd=arr[l][1]*(arr[k][1]-1)*1.0000
                            valueToAdd=valueToAdd/(n*(n-1))
                        else:
                            valueToAdd=2*arr[l][1]*arr[k][1]*1.0000
                            valueToAdd=valueToAdd/(n*(n-1))
                        if diff2==diff1:
                            population2=arr[k][1]+arr[l][1]
                            if population2<=population1:
                                temp[i][j]=temp[i][j]+valueToAdd
                        else:
                            temp[i][j]=temp[i][j]+valueToAdd
                    else:
                        break
    add=arr[:,0];
    temp=np.column_stack((add,temp))
    return temp

def calculateDissimilarityNumerical(x,feature):
    n=np.shape(feature)[0]
    m=np.shape(x)[0]
    temp=np.zeros((n,n))
    featurex=[]
    for i in xrange(0,n):
        for j in xrange(0,m):
            if feature[i]==x[j][0]:
                break
        featurex.append(j);

    for i in xrange(0,n):
        for j in xrange(i,n):
            indexX=featurex[i]
            indexY=featurex[j]
            if feature[i]<feature[j]:
                Dij=x[indexX][indexY+1]
            else:
                Dij=x[indexY][indexX+1]
            lamda=np.log(Dij)
            temp[i][j]=-2*lamda
    return temp

def calculateDissimilarityNominal(x,feature):
    n=np.shape(feature)[0]
    m=np.shape(x)[0]
    temp=np.zeros((n,n))
    featurex=[]
    arr = x[x[:,1].argsort()]

    for i in xrange(0,n):
        for j in xrange(0,m):
            if feature[i]==arr[j][0]:
                break
        featurex.append(j);

    for i in xrange(0,n):
        for j in xrange(i,n):
            if feature[i]!=feature[j]:
                Dij=1
                DijDash=arr[m-1][1]
            else:
                index=featurex[i]
                Dij=arr[index][1]
                if index==0:
                    DijDash=0
                else:
                    DijDash=arr[index-1][1]
                    #to solve problem of matching values
                    if DijDash==Dij:
                        z=index
                        flag=0
                        for k in xrange(0,index-1):
                            z=z-1
                            if Dij==arr[z][1]:
                                continue
                            else:
                                flag=1
                        if flag==1:
                            DijDash= arr[z][1]
                        else:
                            DijDash=0
                    #to solve problem of matching values
                    #remove till here to compare
            # print i," ",j," ",Dij," ",DijDash
            if DijDash==0:
                lamda=1-Dij
            else:
                l=np.log(Dij)
                lDash=np.log(DijDash)
                l=l*Dij
                lDash=lDash*DijDash
                l=l-lDash
                lamda=l*1.0/(Dij-DijDash)
                lamda=1-lamda
            temp[i][j]= 2*lamda
    return temp

a=XNominal[:,0]
# print a.shape
temp1=preprocessing1(a)
DmatNominal=calculateDissimilarityNominal(temp1,a)
lamdaMatrix=DmatNominal
print lamdaMatrix.shape

for i in xrange(1,mNominal):
    a=XNominal[:,i]
    temp1=preprocessing1(a)
    DmatNominal=calculateDissimilarityNominal(temp1,a)
    # print "Nom "
    # print DmatNominal
    lamdaMatrix=np.dstack((lamdaMatrix,DmatNominal))
    print lamdaMatrix.shape

for i in xrange(0,mNumerical):
    b=XNumerical[:,i]
    # print b.shape
    temp2=preprocessing2(b)
    DmatNumerical=calculateDissimilarityNumerical(temp2,b)
    # print "num"
    # print DmatNumerical
    lamdaMatrix=np.dstack((lamdaMatrix,DmatNumerical))
    print lamdaMatrix.shape

lamdaFinal=np.sum(lamdaMatrix,axis = 2)
print lamdaFinal

sum = 0
for k in xrange(0,m-1):
    d=lamdaFinal/2.0
    # print d
    d=d**k
    # print d
    d=d/abs(fact(k))
    # print d
    sum=sum+d
#     sum = sum + (((lamdaFinal/2))**k)/(abs(fact(k)))

disimilarityFinal = np.exp((-1)*(lamdaFinal/2)*sum)
print disimilarityFinal
similarityFinal =  1 - disimilarityFinal

print similarityFinal

# def distance1(a,b):
#     d = 0
#     print a
#     print b
#     for i in xrange(0,len(a)):
#         x = abs(a[i]-b[i])
#         if (a[i]==0):
#             # print a[i]," "
#             a[i]=1
#             # print a[i]," "
#         if (b[i]==0):
#             # print b[i]," "
#             b[i]=1
#             # print b[i]," "
#         # print x
#         d+= pow(x,2)/(a[i]*b[i]*1.0)
#     print d
#     d = np.sqrt(d)
#     # d=d/100
#     return d
#
# def createNovelDistanceMattrix(probabilityMattrix):
#     distanceMattrix = []
#     for i in xrange(0,N):
#         temp = []
#         for j in xrange(0,N):
#             value = distance1(probabilityMattrix[i],probabilityMattrix[j])
#             temp.append(value)
#         distanceMattrix.append(temp)
#     return distanceMattrix
#
# def count(Y,label,number_of_classes):
#     simMattrix = []
#     for i in xrange(0,number_of_classes):
#         temp = []
#         for j in xrange(0,number_of_classes):
#             temp.append(0)
#         simMattrix.append(temp)
#     similarLabels = 0
#     for i in xrange(0,number_of_classes):
#         for j in xrange(0,N):
#             if(Y[j]==i):
#                 x = label[j]
#                 simMattrix[i][x] = simMattrix[i][x] + 1
#         similarLabels+=max(simMattrix[i])
#     return similarLabels/(N*1.0)
#
# NovelDistMatt = createNovelDistanceMattrix(similarityFinal)
#
# kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',n_init=100,tol=0.00001).fit(NovelDistMatt)
# print "---------- SBAC Novel-------------"
# # print kmeans.labels_
# print count(Y,kmeans.labels_,number_of_clusters)
# print "----------------------------"
