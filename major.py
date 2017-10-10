import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
lines = np.loadtxt("fitting_lenses.txt")
print lines.shape
X = lines[:, [1, 2, 3, 4]]
Y = lines[:, [5]]
m = 4
N = 24
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
print correlationCoefficient[0]
# print X
# print " -------------------------------------------------"
# print data.shape;
