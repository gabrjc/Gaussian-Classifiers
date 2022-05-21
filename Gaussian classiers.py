# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:26:12 2022

@author: gabrj
"""

#Gaussian models
#In the first part of this laboratory we will solve the IRIS 
#classication task using Gaussian classiers


import sklearn.datasets
import numpy
import scipy.special
import matplotlib.pyplot as plt

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


#Splitting dataset in training e evaluation part

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:] 
    
    #training
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    #evaluation test
    LTE = L[idxTest]
    DTE = D[:, idxTest]
    
    print(DTR.shape)
    print(LTR.shape)
    
    print(DTE.shape)
    print(LTE.shape)
        
    
    
    return (DTR, LTR), (DTE, LTE)


def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1,v.size))

def covariance_and_mean(D):
    
    mu=vcol(D.mean(1))
    C=numpy.dot(D-mu,(D-mu).T)/float(D.shape[1])
    
    return [C,mu]
    

def logpdf_GAU_ND(X, mu, C):
    
    P=numpy.linalg.inv(C)
    return -0.5*X.shape[0]*numpy.log(numpy.pi*2)+\
        0.5*numpy.linalg.slogdet(P)[1] - 0.5 *\
            (numpy.dot(P,(X-mu))* (X-mu)).sum(0)


def Multivariate_Gaussian_Classifer(h,DTrain, LTrain, DTest, LTest, stamp):
    

    
#Calculate class posterior probability in 3 step.

#1-Calculate loglikelihood (Classic no log) for test sample
#2-Store in a matrix S[i,j] che è la class condition probability 
#per il campione j data la classe i

    SJoint=numpy.zeros((3,DTest.shape[1]))
    logSJoint=numpy.zeros((3,DTest.shape[1]))
    classPriors=[1.0/3.0, 1.0/3.0, 1.0/3.0]

    for label in [0,1,2]:  
        mu,C = h[label]
        SJoint[label,:]=numpy.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[label]
        logSJoint[label,:]=logpdf_GAU_ND(DTest, mu, C).ravel() + numpy.log(classPriors[label])
                        
    SMarginal=SJoint.sum(axis=0)
    logSMarginal=scipy.special.logsumexp(logSJoint,axis=0)
    
    Post1=SJoint / vrow(SMarginal)
    logPost=logSJoint-vrow(logSMarginal)
    Post2=numpy.exp(logPost)
    
    #Trovo la probabilità a posteriori maggiore per i campioni
    LPred1=Post1.argmax(axis=0)
    LPred2=Post2.argmax(axis=0) #for logarithmic
    
    
    res=(LPred1==LTest)
    relerror=((numpy.abs(Post2-Post1))/Post1).max()
    accuracy=(LPred1==LTest).sum()*100/LTest.size
    
    if(stamp==1):
        print("\n\n***********************************\n* Multivariate Gaussian Classifer *\n***********************************\n\n")
        print("Result of assumptions => \n\n",res)
        print("\nRelError between log and classic =",relerror)
        print("Accuracy=",accuracy,"%")
    return accuracy
    
    
    
def Naive_Bayes_Gaussian_Classifer(h,DTrain, LTrain, DTest, LTest, stamp):
    
#Very similar to SVG but i take only the diagonal of covariance matrix

    SJoint=numpy.zeros((3,DTest.shape[1]))
    classPriors=[1.0/3.0, 1.0/3.0, 1.0/3.0]

    for label in [0,1,2]:  
        mu,C = h[label]
        C=C*numpy.identity(C.shape[0])
        SJoint[label,:]=numpy.exp(logpdf_GAU_ND(DTest, mu, C).ravel()) * classPriors[label]
    
    
    SMarginal=SJoint.sum(axis=0)
    
    Post=SJoint / vrow(SMarginal)
    
    #Trovo la probabilità a posteriori maggiore per i campioni
    LPred=Post.argmax(axis=0)


    
    res=(LPred==LTest)
    accuracy=(LPred==LTest).sum()*100/LTest.size
   
    if(stamp==1):
        print("\n\n***********************************\n*Naive Bayes Gaussian Classifiers *\n***********************************\n\n")
        print("Result of assumptions => \n\n",res)
        print("Accuracy=",accuracy,"%")
    return accuracy
    


def Tied_Covariance_Gaussian_Classifer(h, DTrain, LTrain, DTest, LTest, stamp):
    
    Tied=0;
    for label in [0,1,2]:  
        mu,C = h[label]
        Di=DTrain[:,LTrain==label]
        Tied+=Di.shape[1]*C;
    Tied=Tied/DTrain.shape[1]
    Tied=Tied*numpy.identity(Tied.shape[0])
    
    
    
    SJoint=numpy.zeros((3,DTest.shape[1]))
    classPriors=[1.0/3.0, 1.0/3.0, 1.0/3.0]
    
    for label in [0,1,2]:
        mu,C = h[label]
        SJoint[label,:]=numpy.exp(logpdf_GAU_ND(DTest, mu, Tied).ravel()) * classPriors[label]
    
    SMarginal=SJoint.sum(axis=0)
    Post=SJoint / vrow(SMarginal)
    LPred=Post.argmax(axis=0)
    
    
    
    
    res=(LPred==LTest)
    accuracy=(LPred==LTest).sum()*100/LTest.size
    if(stamp==1):
        print("\n\n***************************************\n*Tied Covariance Gaussian Classifiers *\n***************************************\n\n")
        print("Result of assumptions => \n\n",res)
        print("Accuracy=",accuracy,"%")
    
    return accuracy


def Tied_Naive_Bayes(h, DTrain, LTrain, DTest, LTest, stamp):
    
    Tied=0;
    for label in [0,1,2]:  
        mu,C = h[label]
        Di=DTrain[:,LTrain==label]
        Tied+=Di.shape[1]*C;
    Tied=Tied/DTrain.shape[1]
    
    
    
    SJoint=numpy.zeros((3,DTest.shape[1]))
    classPriors=[1.0/3.0, 1.0/3.0, 1.0/3.0]
    
    for label in [0,1,2]:
        mu,C = h[label]
        SJoint[label,:]=numpy.exp(logpdf_GAU_ND(DTest, mu, Tied).ravel()) * classPriors[label]
    
    SMarginal=SJoint.sum(axis=0)
    Post=SJoint / vrow(SMarginal)
    LPred=Post.argmax(axis=0)
    
    
    
    
    res=(LPred==LTest)
    accuracy=(LPred==LTest).sum()*100/LTest.size
    if(stamp==1):
        print("\n\n*******************\n*Tied Naive Bayes *\n*******************\n\n")
        print("Result of assumptions => \n\n",res)
        print("Accuracy=",accuracy,"%")
    
    return accuracy




def Leave_one_out(D, L, Model):
    
    print("\n\n\n\n****************\n*Leave One Out *\n****************\n\n")
    print(Model.__name__)
    
    
    h={}
    GlobalAcc=0;
    for i in range(D.shape[1]):
    #training 
        DTR = numpy.hstack((D[:,0:i],D[:,i+1:]))
        LTR=numpy.hstack((L[0:i], L[i+1:]))
        
#evaluation test
        LTE = L[i:i+1]
        DTE = D[:,i:i+1]
        
        
        for label in [0,1,2]:
            C,mu =covariance_and_mean(DTR[:,LTR==label])    
            h[label]=(mu,C)
        
        GlobalAcc+=Model(h, DTR, LTR, DTE, LTE, 0)
    
    GlobalAcc=GlobalAcc/D.shape[1]
    print("Global Accuracy= ",GlobalAcc, "%")
    
    return GlobalAcc    


if __name__ == '__main__':
    
    D,L=load_iris()
    
    (DTrain, LTrain), (DTest, LTest) = split_db_2to1(D, L)
    
#Calcolo di media empirica e covarianza per ogni classe
    h={}

    for label in [0,1,2]:
        C,mu =covariance_and_mean(DTrain[:,LTrain==label])    
        h[label]=(mu,C)
   
       
#***********************************
#* Multivariate Gaussian Classifer *
#***********************************
    
    Multivariate_Gaussian_Classifer(h,DTrain, LTrain, DTest, LTest,1)
    
    
#**********************************
#* Naive Bayes Gaussian Classifer *
#**********************************
    
    Naive_Bayes_Gaussian_Classifer(h,DTrain, LTrain, DTest, LTest,1)


    
#**************************************
#* Tied Covariance Gaussian Classifer *
#**************************************
    
    Tied_Covariance_Gaussian_Classifer(h,DTrain, LTrain, DTest, LTest,1)

#********************
#* Tied Naive Bayes *
#********************
    
    Tied_Naive_Bayes(h,DTrain, LTrain, DTest, LTest,1)

    
    Leave_one_out(D,L,Multivariate_Gaussian_Classifer)
    Leave_one_out(D,L,Naive_Bayes_Gaussian_Classifer)
    Leave_one_out(D,L,Tied_Covariance_Gaussian_Classifer)
    Leave_one_out(D,L,Tied_Naive_Bayes)

