from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering,FeatureAgglomeration
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
import skfuzzy as fuzz

# has to be changed according to the data set
# lines = np.loadtxt("breast_cancer_data.txt",delimiter=',')
# X = lines[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
# Y = lines[:, [10]]
# m = 9
# N = 699
# number_of_clusters = 2
# for i in xrange(0,N):
#     if(Y[i]==2):
#         Y[i]=0
#     else:
#         Y[i]=1

# lines = np.loadtxt("fitting_lenses.txt")
# X = lines[:, [1, 2, 3, 4]]
# Y = lines[:, [5]]
# m = 4
# N = 24
# number_of_clusters = 3
# for i in xrange(0,N):
#         Y[i]-=1

lines = np.loadtxt("Balloon.txt")
X = lines[:, [0, 1, 2, 3]]
Y = lines[:, [4]]
m = 4
N = 16
number_of_clusters = 2
for i in xrange(0,N):
        Y[i]-=1

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

# lines = np.loadtxt("mushroom.txt")
# X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
# Y = lines[:, [22]]
# m = 22
# N = 8124
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("car.txt")
# X = lines[:, [0, 1, 2, 3, 4, 5]]
# Y = lines[:, [6]]
# m = 6
# N = 1728
# number_of_clusters = 4
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("shuttle-landing-control.txt",delimiter=',')
# X = lines[:, [0, 1, 2, 3, 4, 5]]
# Y = lines[:, [6]]
# m = 6
# N = 15
# number_of_clusters = 2
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("balance-scale.txt",delimiter=',')
# X = lines[:, [1, 2, 3, 4]]
# Y = lines[:, [0]]
# m = 4
# N = 625
# number_of_clusters = 3
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("soybean-large.txt",delimiter=',')
# X = lines[:, [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]
# Y = lines[:, [0]]
# m = 35
# N = 290
# print lines.shape
# number_of_clusters = 15
# for i in xrange(0,N):
#         Y[i]-=1

# lines = np.loadtxt("fisher-order.txt",delimiter=',')
# X = lines[:, [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
# Y = lines[:, [34]]
# print lines.shape
# m = 34
# N = 47
# number_of_clusters = 4
# for i in xrange(0,N):
#         Y[i]-=1

def theta(a,b):
    similarity = 0;
    for i in xrange(0,m):
        if(a[i]==b[i]):
            similarity = similarity + 1
    return similarity

def dist(a,b):
    d = 0;
    for i in xrange(0,m):
        if(a[i]!=b[i]):
            d = d + 1
    return d

def distance1(a,b):
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

def createData():
    data = []
    for i in xrange(0,N):
        # print "data ",i
        temp = []
        for j in xrange(0,N):
            temp.append((theta(X[i],X[j])*(1.0))/m)
        data.append(temp)
    return data

def createDistanceMattrix1(X):
    distanceMattrix = []
    for i in xrange(0,N):
        # print "distance ",i
        temp = []
        for j in xrange(0,N):
            temp.append(dist(X[i],X[j]))
        distanceMattrix.append(temp)
    return distanceMattrix

def createNovelDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        # print "novel ",i
        temp = []
        for j in xrange(0,N):
            value = distance1(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

def createEuclideanDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        # print "euclidean ",i
        temp = []
        for j in xrange(0,N):
            value = distance.euclidean(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix

def createCosineDistanceMattrix(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        # print "cosine ",i
        temp = []
        for j in xrange(0,N):
            value = distance.cosine(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix
data = createData()
distMatt1 = createDistanceMattrix1(X)
NovelDistMatt = createNovelDistanceMattrix(data)
EuclideanDistMatt = createEuclideanDistanceMattrix(data)
CosineDistMatt = createCosineDistanceMattrix(data)
# print data
# print distMatt1
# print distMatt2
a = sum(distMatt1,[])
b = sum(NovelDistMatt,[])
correlationCoefficient = pearsonr(a,b)
# print "correlationCoefficient : ",correlationCoefficient[0]

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

kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',n_init=100,tol=0.00001).fit(NovelDistMatt)
print "---------- SBC Novel-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(EuclideanDistMatt)
print "---------- SBC Euclidean-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(CosineDistMatt)
print "---------- SBC Cosine-------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

km = KModes(n_clusters=number_of_clusters, n_init=100).fit(X)
print "---------- kmodes ----------"
# print km.labels_
print count(Y,km.labels_,number_of_clusters)
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
