from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering,FeatureAgglomeration
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
import skfuzzy as fuzz

# has to be changed according to the data set
#X==testing data
#Y==Resultant classes

lines = np.loadtxt("breast_cancer_data.txt",delimiter=' ')
X = lines[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
Y = lines[:, [10]]
m = 9
N = 699
number_of_clusters = 2
for i in xrange(0,N):
    if(Y[i]==2):
        Y[i]=0
    else:
        Y[i]=1

# lines = np.loadtxt("fitting_lenses.txt")
# X = lines[:, [1, 2, 3, 4]]
# Y = lines[:, [5]]
# m = 4
# N = 24
# number_of_clusters = 3
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Balloon.txt")
# X = lines[:, [0, 1, 2, 3]]
# Y = lines[:, [4]]
# m = 4
# N = 16
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("SoyBeenSmall.txt")
# X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]
# Y = lines[:, [35]]
# m = 35
# N = 47
# number_of_clusters = 4
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("HayesRoth.txt")
# X = lines[:, [1, 2, 3, 4]]
# Y = lines[:, [5]]
# m = 4
# N = 132
# number_of_clusters = 3
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Promoters.txt")
# X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,
# 44,45,46,47,48,49,50,51,52,53,54,55,56]]
# Y = lines[:, [57]]
# m = 57
# N = 106
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Monks.txt")
# X = lines[:, [0,1, 2, 3, 4,5]]
# Y = lines[:, [6]]
# m = 6
# N = 432
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("nursery.txt")
# X = lines[:, [0, 1, 2, 3, 4, 5, 6, 7]]
# Y = lines[:, [8]]
# m = 8
# N = 12959
# number_of_clusters = 5
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("Voters.txt")
# X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15]]
# Y = lines[:, [16]]
# m = 16
# N = 435
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

lines = np.loadtxt("mushroom.txt")
X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
Y = lines[:, [22]]
m = 22
N = 8124
number_of_clusters = 2
for i in xrange(0,N):
        Y[i]-=1

#meassure AC
# def count(Y,label,number_of_classes):
#     simMattrix = []
#     for i in xrange(0,number_of_classes):
#         temp = []
#         for j in xrange(0,number_of_classes):
#             temp.append(0)
#         simMattrix.append(temp)
#     similarLabels = 0
#     for j in xrange(0,N):
#         x = label[j]
#         y = Y[j]
#         simMattrix[x][int(y[0])] = simMattrix[x][int(y[0])] + 1
#     #print simMattrix
#     for j in xrange(0,number_of_classes):
#         similarLabels+=max(simMattrix[j])
#     #print similarLabels
#     return similarLabels/(N*1.0)

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

#measure error(no of wrong results)
# def error(Y,label):
#     e=0;
#     for i in xrange(0,N):
#         if(Y[i]!=label[i]):
#             e=e+1;
#     return e/(N*1.0)

#find the frequency of a value of a particular attribute
def count1(i,x):
    c=0;
    for j in xrange(0,N):
        if(X[j][i]==x):
            c=c+1
    return c

#find the size of domain of attributes
M=X.max(0);
#print M

#find the probability of value of attributes matching in any two chosen objects
def psOfAr(i):
    pst=0;
    for j in xrange(1,int(M[i])):
        cj=count1(i,j);
        #print cj
        pst=pst+((cj)*(cj-1));
    ps=pst/(N*(N-1)*1.0);
    #print ps
    return ps;

#return similarity between two objects
def theta(a,b):
    similarity = 0;
    for i in xrange(0,m):
        if(a[i]==b[i]):
            similarity = similarity + 1
    return similarity

#create similarity matrix(probability of similarity:similarity by M)
def createData():
    data = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            temp.append((theta(X[i],X[j])*(1.0))/m)
        data.append(temp)
    return data

#get ps for all atributes
# ps=[];
# for i in xrange(0,m):
#     ps.append(psOfAr(i));

#01 distance
def zeroOneDistance(a,b):
    d = 0;
    for i in xrange(0,m):
        if(a[i]!=b[i]):
            d = d + 1
    return d

#different distancemetric with weight
def diffDistance(a,b):
    d = 0;
    totalWeight=0;
    for i in xrange(0,m):
        if(a[i]!=b[i]):
            ca=count1(i,a[i])
            cb=count1(i,b[i])
            # print ca
            # print cb
            distance=N*((ca*(ca-1))+(cb*(cb-1))/(m*(m-1)*1.0));
            d=d+ps[i]*distance;
            totalWeight=totalWeight+ps[i];
        else:
            totalWeight=totalWeight+(1-ps[i]);
    d=d/(totalWeight*1.0);
    return d

#different distancemetric without weight
def diffDistanceWithoutWeight(a,b):
    d = 0;
    totalWeight=0;
    for i in xrange(0,m):
        if(a[i]!=b[i]):
            ca=count1(i,a[i])
            cb=count1(i,b[i])
            # print ca
            # print cb
            distance=N*((ca*(ca-1))+(cb*(cb-1))/(m*(m-1)*1.0));
            d=d+distance;
    return d

#novel distance metric
def novelDistance(a,b):
    d = 0
    for i in xrange(0,len(a)):
        x = abs(a[i]-b[i])
        if (a[i]==0):
            # print a[i]," "
            a[i]=1
            # print a[i]," "
        if (b[i]==0):
            # print b[i]," "
            b[i]=1
            # print b[i]," "
        # print x
        d+= pow(x,2)/(a[i]*b[i]*1.0)
    d = np.sqrt(d)
    return d

#novel distance metric with weight
# def novelDistanceWithWeight(a,b):
#     print a
#     print b
#     d = 0
#     totalWeight=0
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
#             distance= pow(x,2)/(a[i]*b[i]*1.0)
#             d=d+ps[i]*distance;
#             totalWeight=totalWeight+ps[i];
#         else:
#             totalWeight=totalWeight+(1-ps[i]);
#     d=d/(totalWeight*1.0);
#     d = np.sqrt(d)
#     return d

#create noveldistance matrix
def createDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = novelDistance(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

#create noveldistance matrix with weight
# def createDistanceMattrix0(probabilityMattrix):
#     distanceMattrix = []
#     for i in xrange(0,N):
#         temp = []
#         for j in xrange(0,N):
#             temp.append(novelDistanceWithWeight(probabilityMattrix[i],probabilityMattrix[j]))
#         distanceMattrix.append(temp)
#     return distanceMattrix

#create diffdistance matrix
def createDistanceMattrix1(X):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            temp.append(diffDistance(X[i],X[j]))
        distanceMattrix.append(temp)
    return distanceMattrix

#create diffdistance matrix without weight
def createDistanceMattrix1_1(X):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            temp.append(diffDistanceWithoutWeight(X[i],X[j]))
        distanceMattrix.append(temp)
    return distanceMattrix

#basic 01distance matrix
def createDistanceMattrix2(X):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            temp.append(zeroOneDistance(X[i],X[j]))
        distanceMattrix.append(temp)
    return distanceMattrix

#spacial structure distance matrix with euclidean distance
def createDistanceMattrix3(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance.euclidean(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

#spacial structure distance matrix with cosine distance
def createDistanceMattrix4(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance.cosine(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

#data is the similarity matrix
data = createData()



# # distMatt0 = createDistanceMattrix0(data)
distMatt = createDistanceMattrix(data)
# # distMatt1 = createDistanceMattrix1(X)
# # distMatt1_1 = createDistanceMattrix1_1(X)
# # distMatt2 = createDistanceMattrix2(X)
distMatt3 = createDistanceMattrix3(data)
distMatt4 = createDistanceMattrix4(data)

# print "---------- Distance Matrix-------------"
# print distMatt1

#result of kmeans on distance matrix
# kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt0)
# print "---------- SBC with novel distance weighted-------------"
# # print kmeans.labels_
# print count(Y,kmeans.labels_,number_of_clusters)
# print "----------------------------"
kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt)
print "---------- SBC with novel distance-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"
# kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt1)
# print "---------- diff distance -------------"
# # print kmeans.labels_
# print count(Y,kmeans.labels_,number_of_clusters)
# print "----------------------------"
# kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt1_1)
# print "---------- diff distance without weight-------------"
# # print kmeans.labels_
# print count(Y,kmeans.labels_,number_of_clusters)
# print "----------------------------"
# kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt2)
# print "---------- silly 01 distance -------------"
# # print kmeans.labels_
# print count(Y,kmeans.labels_,number_of_clusters)
# print "----------------------------"
kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt3)
print "---------- SBC with euclidean distance -------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"
kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(distMatt4)
print "---------- SBC with cosine distance -------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(distMatt3), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
print "---------- Fuzzy SBC with euclidean distance -------------"
# print kmeans.labels_
print count(Y,cluster_membership,number_of_clusters)
print "----------------------------"
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(distMatt4), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
print "---------- Fuzzy SBC with cosine distance -------------"
# print kmeans.labels_
print count(Y,cluster_membership,number_of_clusters)
print "----------------------------"
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(distMatt), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)
print "---------- Fuzzy SBC with novel distance -------------"
# print kmeans.labels_
print count(Y,cluster_membership,number_of_clusters)
print "----------------------------"
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         np.transpose(distMatt1_1), number_of_clusters, 2, error=0.005, maxiter=1000, init=None)
# cluster_membership = np.argmax(u, axis=0)
# print "---------- Fuzzy SBC with different distance -------------"
# # print kmeans.labels_
# print count(Y,cluster_membership,number_of_clusters)
# print "----------------------------"

# # result of kmodes on simple data
km = KModes(n_clusters=number_of_clusters, n_init=100).fit(X)
print "---------- kmodes ----------"
# print km.labels_
print count(Y,km.labels_,number_of_clusters)
print "----------------------------"

# bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=N)
# ms = MeanShift(bandwidth=bandwidth).fit(data)
# print "---------- Mean Shift ----------"
# print ms.labels_
# print count(Y,ms.labels_,number_of_clusters)
# print "----------------------------"

fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(distMatt)
print "---------- Hierarchical with novelDistance ----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(distMatt3)
print "---------- Hierarchical with euclideanDistance ----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(distMatt4)
print "---------- Hierarchical with cosineDistance ----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
