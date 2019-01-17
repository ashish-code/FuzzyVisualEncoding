'''
Created on 19 Jan 2012

@author: ag00087
'''

import numpy as np
from optparse import OptionParser
import sys
import matplotlib.pyplot as plt
#global paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
resultDir = '/Fuzzy/Result/'
outputDir = '/vol/vssp/diplecs/ash/Expt/FuzzyClustering/icip2012/'

datasets = ['Caltech101','VOC2006','VOC2010','Caltech256','Scene15']

dim = 2
methods = ['Kmeans','FCM','GK']
words = [16,32,64,128,256,512]
#acquire program arguments
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',default='VOC2006',metavar='dataset',help='visual dataset')
parser.add_option('-w','--word',action='store',type='int',dest='word',default=16,help=' number of words in the dictionary')
#global variables
catidfname = 'catidlist.txt' # list of categories in the dataset

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def plotfuzzyresultCategory():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    #dataset = options.dataset
    #word = options.word
    word = 16
    #catmap = getCatMap(dataset)
    #catList = catmap.keys()
    #nCategory = len(catList)
    accResult = np.zeros((3,len(datasets)),np.float)
    accResultStd = np.zeros((3,len(datasets)),np.float)
    for i,method in enumerate(methods):
        for j,dataset in enumerate(datasets):
            accFileName = rootDir+dataset+resultDir+method+str(dim)+str(word)+'.acc'
            try:
                accVal = np.loadtxt(accFileName, dtype=np.float, delimiter=' ')
                accResult[i,j] = np.mean(accVal)
                accResultStd[i,j] = np.std(accVal)
            except:
                pass
        
        pass
    print accResult
    ind = np.arange(len(datasets))  # the x locations for the groups
    width = 0.20       # the width of the bars
    maxacc = np.max(accResult)+0.05
    minacc = np.min(accResult)-0.05
    nXTicks = len(datasets)
    fig = plt.figure()
    #ax = plt.subplot(111)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, accResult[0,:], width, color='y',label='BoF')
    rects2 = ax.bar(ind+width, accResult[1,:], width, color='g',label='FCM')
    rects3 = ax.bar(ind+width+width, accResult[2,:], width, color='k',label='GK')
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('BoF', 'FCM', 'GK') )
    ax.set_xticklabels(datasets,rotation=30,size='medium',ha='center')
    
    ax.set_xticks(ind+width)
    

    outPath = outputDir+'perfDatasets'+str(word)+'.png'
    
    #plt.plot(np.arange(1,len(datasets)+1),accResult[2,:],'r-s',linewidth=2.5,label='GK')
    #plt.plot(np.arange(1,len(datasets)+1),accResult[1,:],'k--d',linewidth=2.5,label='FCM')
    #plt.plot(np.arange(1,len(datasets)+1),accResult[0,:],'b-.o',linewidth=2.5,label='BoF')
    
    plt.xlabel('visual data set',size='large')
    plt.ylabel('mAcc',size='large')
#    plt.title('%s Performance: %s ' % (title,dataset))
    #plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    plt.ylim([minacc,maxacc])
    #ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(datasets,rotation=0,size='medium',ha='center')
    plt.savefig(outPath,format='png')
    plt.show()
    pass

if __name__=='__main__':
    plotfuzzyresultCategory()
    pass