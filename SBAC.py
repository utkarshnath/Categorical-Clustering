from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering,FeatureAgglomeration
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
import skfuzzy as fuzz
import numpy as np
from scipy.misc import factorial as fact
import random

lines = np.loadtxt("breast_cancer_data.txt",delimiter=',')
XNominal = lines[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
XNumerical=[]
Y = lines[:, [10]]
m = 9
mNominal=9
mNumerical=0
N = 699
number_of_clusters = 2
for i in xrange(0,N):
    if(Y[i]==2):
        Y[i]=0
    else:
        Y[i]=1

# lines = np.loadtxt("fitting_lenses.txt")
# XNominal = lines[:, [1, 2, 3, 4]]
# XNumerical=[]
# Y = lines[:, [5]]
# m = 4
# mNominal=4
# mNumerical=0
# N = 24
# number_of_clusters = 3
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("HeartDiseaseData.txt")
# XNominal = lines[:, [1, 2, 5, 6, 8, 10,11,12]]
# XNumerical = lines[:, [0, 3, 4, 7, 9]]
# mNominal=8
# mNumerical=5
# Y = lines[:, [13]]
# m = 13
# N = 270
# number_of_clusters = 7
# for i in xrange(0,N):
#         Y[i]-=1

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

# lines = np.loadtxt("SoyBeenSmall.txt")
# XNominal = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]
# XNumerical=[]
# Y = lines[:, [35]]
# m = 35
# mNominal=35
# mNumerical=0
# N = 47
# number_of_clusters = 4
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("HayesRoth.txt")
# XNominal = lines[:, [1, 2, 3, 4]]
# XNumerical=[]
# Y = lines[:, [5]]
# m = 4
# mNominal=4
# mNumerical=0
# N = 132
# number_of_clusters = 3
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Promoters.txt")
# XNominal = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,
# 44,45,46,47,48,49,50,51,52,53,54,55,56]]
# XNumerical=[]
# Y = lines[:, [57]]
# m = 57
# mNominal=57
# mNumerical=0
# N = 106
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Monks.txt")
# XNominal = lines[:, [0,1, 2, 3, 4,5]]
# XNumerical=[]
# Y = lines[:, [6]]
# m = 6
# mNominal=6
# mNumerical=0
# N = 432
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("nursery.txt")
# XNominal = lines[:, [0, 1, 2, 3, 4, 5, 6, 7]]
# XNumerical=[]
# Y = lines[:, [8]]
# m = 8
# mNominal=8
# mNumerical=0
# N = 12959
# number_of_clusters = 5
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Voters.txt")
# XNominal = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15]]
# XNumerical=[]
# Y = lines[:, [16]]
# m = 16
# mNominal=16
# mNumerical=0
# N = 435
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("ticTacToe.txt")
# XNominal = lines[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
# XNumerical=[]
# Y = lines[:, [9]]
# m = 9
# mNominal=9
# mNumerical=0
# N = 958
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("spectf.txt")
# XNominal = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
# 33,34,35,36,37,38,39,40,41,42,43]]
# Y = lines[:, [44]]
# m = 44
# mNominal=44
# XNumerical=[]
# mNumerical=0
# N = 267
# number_of_clusters = 2

# lines = np.loadtxt("Audiology.txt")
# X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,
# 44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]]
# Y = lines[:, [69]]
# m = 69
# mNominal=69
# XNumerical=[]
# mNumerical=0
# N = 226
# number_of_clusters = 24
# for i in xrange(0,N):
#         Y[i]-=1
#
# lines = np.loadtxt("ZooD.txt")
# XNominal = lines[:, [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,14,15,16]]
# Y = lines[:, [0]]
# XNumerical = lines[:, [13]]
# m = 16
# mNominal=15
# mNumerical=1
# N = 101
# number_of_clusters = 7
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Flowies.txt")
# XNominal = lines[:, [1, 2, 3, 4]]
# Y = lines[:, [0]]
# XNumerical = lines[:, [5]]
# mNominal = 4
# mNumerical=1
# m=5
# N = 168
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("HorseShoeCrab.txt")
# XNominal = lines[:, [0, 1]]
# Y = lines[:, [5]]
# XNumerical = lines[:, [2, 4]]
# m = 4
# mNominal=2
# mNumerical=2
# N = 173
# number_of_clusters = 2

# lines = np.loadtxt("mushroom.txt")
# Class0=lines[lines[:, 22] == 1]
# Class1=lines[lines[:, 22] == 2]
# start0=random.randint(1,np.shape(Class0)[1])
# start1=random.randint(1,np.shape(Class1)[1])
# Class0=Class0[start0:start0+199,:]
# Class1=Class1[start1:start1+199,:]
# reducedData=np.concatenate((Class0,Class1), axis=0);
# XNominal = reducedData[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
# Y = reducedData[:, [22]]
# mNominal = 22
# mNumerical=0
# m=22
# N = 200
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1




# a = np.array( [0,1,1,0,2,2,2,2,0,1] )
# b = np.array( [6,5.5,7.5,6,10.5,9,7.5,9, 7.5,7.5] )


###################### Preprossesing for nominal features ######################
# Here we have calculated dis-similarity score of feature to itself. eg score of (a,a) , (b,b)
# Different features would have dis-similarity score equal to 1. eg (a,b)
# Stored the dis-similarity in temp(Shape = feature domain X 2)

def preprocessingNominal(values):
    unique_elements, counts_elements = np.unique(values, return_counts=True)
    elementsAndFrequencies=np.column_stack((unique_elements,counts_elements))
    arr = elementsAndFrequencies[elementsAndFrequencies[:,1].argsort()]
    n = np.shape(values)[0]
    rows = np.shape(arr)[0]
    previousVal = 0
    j = 0
    breakFlag=1
    temp = []
    for i in xrange(0,rows):
        if (j>i) or ( j==rows-1 and breakFlag==0):
            temp.append(previousVal)
            continue
        sameFrequencyCount=1;
        breakFlag=0
        for j in xrange(i+1,rows):
            if(arr[j][1]==arr[i][1]):
                sameFrequencyCount=sameFrequencyCount+1
            else:
                breakFlag=1
                break
        val = arr[i][1] * ((arr[i][1])-1)
        val = val * 1.0/(n*(n-1)*1.0)
        val = val * sameFrequencyCount
        val = val + previousVal
        temp.append(val)
        previousVal = val
    add = arr[:,0]
    temp = np.column_stack((add,temp))
    return temp

################################################################################

################# CALCULATE LAMDA FOR NOMINAL FEATURES #########################
# Here we are finding just lower dis-similarity (DijDash) and are then appling
# formula to calculate lamda.
# Special case - When no just smaller dis-similarity available i.e min dis-similarity,
# here we have taken DijDash = 0

def calculateLamdaNominal(x,values):
    n = np.shape(values)[0]
    m = np.shape(x)[0]
    lamdaNominal = np.zeros((n,n))
    mappingOfValues = []
    arr = x[x[:,1].argsort()]

    for i in xrange(0,n):
        for j in xrange(0,m):
            if values[i] == arr[j][0]:
                break
        mappingOfValues.append(j);

    for i in xrange(0,n):
        for j in xrange(i,n):
            if values[i]!=values[j]:
                Dij = 1
                DijDash = arr[m-1][1]
            else:
                index = mappingOfValues[i]
                Dij = arr[index][1]
                if index==0:
                    DijDash=0
                else:
                    DijDash=arr[index-1][1]
                    if DijDash==Dij:
                        z=index
                        flag=0
                        for k in xrange(0,index-1):
                            z=z-1
                            if Dij==arr[z][1]:
                                continue
                            else:
                                flag=1
                                break
                        if flag==1:
                            DijDash= arr[z][1]
                        else:
                            DijDash=0
            if DijDash==0:
                lamda = 1-np.log(Dij)
            else:
                l = np.log(Dij)
                lDash = np.log(DijDash)
                l = l*Dij
                lDash = lDash*DijDash
                l = l-lDash
                lamda = l*1.0/(Dij-DijDash)
                lamda = 1-lamda
            lamdaNominal[i][j] = 2*lamda
    return lamdaNominal

################################################################################

################### PREPROCESSING FOR NUMERICAL FEATURES #######################
# Here we have calculated dis-similarity score for Numerical features, (N X N)
# and stored that in temp

def preprocessingNumerical(values):
    n=np.shape(values)[0];
    unique_elements, counts_elements = np.unique(values, return_counts=True)
    elementsAndFrequencies = np.column_stack((unique_elements,counts_elements))
    arr = elementsAndFrequencies[elementsAndFrequencies[:,0].argsort()]
    rows = np.shape(arr)[0];
    frequencyMatrix = np.zeros((rows,rows))
    for i in xrange(0,rows):
        for j in xrange(i,rows):
            if i==j:
                frequencyMatrix[i][j] = arr[i][1];
            else:
                frequencyMatrix[i][j] = frequencyMatrix[i][j-1]+arr[j][1];
    temp = np.zeros((rows,rows))
    for i in xrange(0,rows):
        for j in xrange(i,rows):
            for k in xrange(0,rows):
                for l in xrange(k,rows):
                    diff1 = abs(arr[j][0]-arr[i][0])
                    diff2 = abs(arr[l][0]-arr[k][0])
                    population1 = frequencyMatrix[max(i,j)][min(i,j)]
                    population2 = frequencyMatrix[max(k,l)][min(k,l)]
                    if diff2<=diff1:
                        if diff2==0:
                            valueToAdd = arr[l][1] * (arr[l][1]-1) * 1.0
                            valueToAdd = valueToAdd/(n*(n-1))
                        else:
                            valueToAdd = 2 * arr[l][1] * arr[k][1] * 1.0
                            valueToAdd = valueToAdd/(n*(n-1))
                        if diff2==diff1:
                            if population2<=population1:
                                temp[i][j] = temp[i][j] + valueToAdd
                        else:
                            temp[i][j] = temp[i][j]+valueToAdd
                    else:
                        break
    index = arr[:,0];
    temp = np.column_stack((index,temp))
    return temp

################################################################################

################# CALCULATE LAMDA FOR NOMINAL FEATURES #########################

def calculateLamdaNumerical(x,feature):
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
            if Dij==0:
                lamda = 0
            else:
                lamda=np.log(Dij)
            temp[i][j]=-2*lamda
    return temp

################################################################################


values = XNominal[:,0]
temp1 = preprocessingNominal(values)
DmatNominal = calculateLamdaNominal(temp1,values)
lamdaMatrix = DmatNominal

for i in xrange(1,mNominal):
    values = XNominal[:,i]
    temp1 = preprocessingNominal(values)
    DmatNominal = calculateLamdaNominal(temp1,values)
    lamdaMatrix = np.dstack((lamdaMatrix,DmatNominal))


for i in xrange(0,mNumerical):
    values = XNumerical[:,i]
    temp2 = preprocessingNumerical(values)
    DmatNumerical = calculateLamdaNumerical(temp2,values)
    lamdaMatrix = np.dstack((lamdaMatrix,DmatNumerical))

# print lamdaMatrix
lamdaFinal=np.sum(lamdaMatrix,axis = 2)
# print "lamda final ",lamdaFinal
#
sum = 0
for k in xrange(0,m-1):
    d=lamdaFinal/2.0
    d=d**k
    d=d/(fact(k))
    sum=sum+d


disimilarityFinal = np.exp(-1*(lamdaFinal/2))*sum
similarityFinal =  1 - disimilarityFinal


def distance1(a,b):
    d = 0
    for i in xrange(0,len(a)):
        x = abs(a[i]-b[i])
        if (a[i]==0):
            a[i]=1
        if (b[i]==0):
            b[i]=1
        d+= pow(x,2)/(a[i]*b[i]*1.0)

    d = np.sqrt(d)
    return d

def createNovelDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance1(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

def createEuclideanDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance.euclidean(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

def createManhattanDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance.cityblock(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

def createCosineDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance.cosine(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

def count(Y,label,number_of_classes):
    simMattrix = []
    for i in xrange(0,number_of_classes):
        temp = []
        for j in xrange(0,number_of_classes):
            temp.append(0)
        simMattrix.append(temp)
    similarLabels = 0
    for i in xrange(0,number_of_classes):
        for j in xrange(0,N):
            if(Y[j]==i):
                x = label[j]
                simMattrix[i][x] = simMattrix[i][x] + 1
        similarLabels+=max(simMattrix[i])
    return similarLabels/(N*1.0)

NovelDistMatt = createNovelDistanceMattrix(similarityFinal)
EuclideanDistMatt = createEuclideanDistanceMattrix(similarityFinal)
ManhattanDistMatt = createManhattanDistanceMattrix(similarityFinal)
CosineDistMatt = createCosineDistanceMattrix(similarityFinal)

kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',n_init=100,tol=0.00001).fit(NovelDistMatt)
print "---------- SBAC Novel-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',n_init=100,tol=0.00001).fit(ManhattanDistMatt)
print "---------- SBC Manhattan-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',n_init=100,tol=0.00001).fit(EuclideanDistMatt)
print "---------- SBC Euclidean-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',n_init=100,tol=0.00001).fit(CosineDistMatt)
print "---------- SBC Cosine-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(NovelDistMatt), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
print "---------- Fuzzy SBC Novel-------------"
# print kmeans.labels_
print count(Y,cluster_membership,number_of_clusters)
print "----------------------------"
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(EuclideanDistMatt), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
print "---------- Fuzzy SBC Euclidean-------------"
# print kmeans.labels_
print count(Y,cluster_membership,number_of_clusters)
print "----------------------------"
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(CosineDistMatt), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
print "---------- Fuzzy SBC Cosine -------------"
# print kmeans.labels_
print count(Y,cluster_membership,number_of_clusters)
print "----------------------------"

# bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=N)
# ms = MeanShift(bandwidth=bandwidth).fit(data)
# print "---------- Mean Shift ----------"
# print ms.labels_
# print count(Y,ms.labels_,number_of_clusters)
# print "----------------------------"

fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(NovelDistMatt)
print "---------- Hierarchical Novel----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(EuclideanDistMatt)
print "---------- Hierarchical Euclidean ----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(CosineDistMatt)
print "---------- Hierarchical Cosine----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
