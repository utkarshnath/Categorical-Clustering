from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering,FeatureAgglomeration
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
import skfuzzy as fuzz

lines = np.loadtxt("breast_cancer_data.txt",delimiter=',')
X = lines[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
Y = lines[:, [10]]
m = 9
N = 699
number_of_clusters = 2

# lines = np.loadtxt("fitting_lenses.txt")
# X = lines[:, [1, 2, 3, 4]]
# Y = lines[:, [5]]
# m = 4
# N = 24
# number_of_clusters = 3


for i in xrange(0,N):
    if(Y[i]==2):
        Y[i]=0
    else:
        Y[i]=1
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

def createData():
    data = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            temp.append(theta(X[i],X[j])*(1.0)/m)
        data.append(temp)
    return data

def createDistanceMattrix1(X):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            temp.append(dist(X[i],X[j]))
        distanceMattrix.append(temp)
    return distanceMattrix

def createDistanceMattrix2(probabilityMattrix):
    distanceMattrix = []
    for i in xrange(0,N):
        temp = []
        for j in xrange(0,N):
            value = distance.euclidean(probabilityMattrix[i],probabilityMattrix[j])
            temp.append(value)
        distanceMattrix.append(temp)
    return distanceMattrix
data = createData()
distMatt1 = createDistanceMattrix1(X)
distMatt2 = createDistanceMattrix2(data)
a = sum(distMatt1,[])
b = sum(distMatt2,[])
correlationCoefficient = pearsonr(a,b)
print "correlationCoefficient : ",correlationCoefficient[0]

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

kmeans = KMeans(n_clusters=number_of_clusters,n_init=100,tol=0.00001).fit(data)
print "---------- SBC -------------"
# print kmeans.labels_
print count(Y,kmeans.labels_,number_of_clusters)
print "----------------------------"

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

fa = FeatureAgglomeration(n_clusters=number_of_clusters).fit(data)
print "---------- Hierarchical  ----------"
# print fa.labels_
print count(Y,fa.labels_,number_of_clusters)
print "----------------------------"
