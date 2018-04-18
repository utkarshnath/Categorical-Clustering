
# # feature=array.array('B',['a','c','a','c','b','b','a','c','b'])
# feature= array.array('i',[1, 2, 3, 1, 5])
# def preprocessing(feature):
#     df=pd.DataFrame(feature)
#     return df.unique()
#
# print preprocessing(feature)


import numpy as np

a = np.array( [10,10,20,10,20,20,20,30, 30,50] )
b = np.array( [6,5.5,7.5,6,10.5,9,7.5,9, 7.5,7.5] )

def preprocessing1(a):
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    elementsAndFrequencies=np.column_stack((unique_elements,counts_elements))
    print elementsAndFrequencies
    arr = elementsAndFrequencies[elementsAndFrequencies[:,1].argsort()]
    print arr
    n=10;
    rows=np.shape(arr)[0];
    previousVal=0;
    j=0;
    temp=[];
    for i in xrange(0,rows):
        if(j>i):
            print "continuing same"
            temp.append(previousVal)
            continue
        sameFrequencyCount=1;
        for j in xrange(i+1,rows):
            print "counting for same"
            if(arr[j][1]==arr[i][1]):
                sameFrequencyCount=sameFrequencyCount+1
                print "counting for same ",sameFrequencyCount
            else:
                print "breaking"
                break
        val=arr[i][1]* ((arr[i][1])-1)  ;
        print "calculating val ",val
        val=val*1.0/(n*(n-1)*1.0);
        print "calculating val ",val
        val=val* sameFrequencyCount
        print "calculating val ",val
        val=val+previousVal
        print "calculated val ",val
        temp.append(val)
        previousVal=val;

    add=arr[:,0]
    print add
    temp=np.column_stack((add,temp))
    print temp
    return temp

def preprocessing2(b):
    n=10
    unique_elements, counts_elements = np.unique(b, return_counts=True)
    elementsAndFrequencies=np.column_stack((unique_elements,counts_elements))
    print elementsAndFrequencies
    arr = elementsAndFrequencies[elementsAndFrequencies[:,0].argsort()]
    print arr
    rows=np.shape(arr)[0];
    frequencyMatrix=np.zeros((rows,rows))
    for i in xrange(0,rows):
        for j in xrange(i,rows):
            if i==j:
                frequencyMatrix[i][j]=arr[i][1];
            else:
                frequencyMatrix[i][j]=frequencyMatrix[i][j-1]+arr[j][1];
    print frequencyMatrix
    temp=np.zeros((rows,rows))
    for i in xrange(0,rows):
        for j in xrange(i,rows):
            for k in xrange(0,rows):
                for l in xrange(k,rows):
                    diff1=(arr[j][0]-arr[i][0])
                    diff2=(arr[l][0]-arr[k][0])
                    population1=arr[j][1]+arr[i][1]
                    print "diff1 ",diff1
                    print "diff2 ",diff2
                    if diff2<=diff1:
                        if diff2==0:
                            valueToAdd=arr[l][1]*(arr[k][1]-1)*1.0000
                            print valueToAdd
                            valueToAdd=valueToAdd/(n*(n-1))
                            print valueToAdd
                        else:
                            valueToAdd=2*arr[l][1]*arr[k][1]*1.0000
                            print valueToAdd
                            valueToAdd=valueToAdd/(n*(n-1))
                            print valueToAdd
                        if diff2==diff1:
                            population2=arr[k][1]+arr[l][1]
                            if population2<=population1:
                                print "diff equal and adding ",l," ",k," ",valueToAdd
                                temp[i][j]=temp[i][j]+valueToAdd
                        else:
                            print "diff less and adding ",l," ",k," ",valueToAdd
                            temp[i][j]=temp[i][j]+valueToAdd
                    else:
                        break
            print "value ",i," ",j," ",temp[i][j]
            print "....................."
            print "....................."
            print "....................."
            print "....................."
            print "....................."

    add=arr[:,0];
    print add
    temp=np.column_stack((add,temp))
    print temp
    return temp

def calculateDissimilarityNumerical(x,feature):
    # print x
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
            print Dij
            lamda=np.log(Dij)
            temp[i][j]=-2*lamda
    print "temp Numerical ",temp
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
                DijDash=arr[0][1]
            else:
                index=featurex[i]
                Dij=arr[index][1]
                if index==m:
                    DijDash=0
                else:
                    DijDash=arr[index-1][1]
            print Dij
            print DijDash
            l=np.log(Dij)
            lDash=np.log(DijDash)
            l=l*Dij
            lDash=lDash*DijDash
            l=l-lDash
            lamda=l*1.0/(Dij-DijDash)
            lamda=1-lamda
            temp[i][j]= 2*lamda
    print "temp Nominal ",temp
    return temp

temp1=preprocessing1(a)
temp2=preprocessing2(b)
DmatNumerical=calculateDissimilarityNumerical(temp2,b)
DmatNominal=calculateDissimilarityNominal(temp1,a)

lamdaMatrix=np.dstack((DmatNumerical,DmatNominal))
lamdaFinal=np.sum(lamdaMatrix,axis = 2)
print lamdaFinal
