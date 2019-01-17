'''
Created on 19 Jan 2012
plot the mean accuracy for different categories of a dataset
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
    dataset = options.dataset
    word = options.word
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    nCategory = len(catList)
    accResult = np.zeros((3,nCategory),np.float)
    for i,method in enumerate(methods):
        accFileName = rootDir+dataset+resultDir+method+str(dim)+str(word)+'.acc'
        try:
            accVal = np.loadtxt(accFileName, dtype=np.float, delimiter=' ')
            accResult[i,:] = accVal
        except:
            pass
        
        pass
    print accResult
    
    maxacc = np.max(accResult)+0.05
    minacc = np.min(accResult)-0.05
    
    
    outPath = outputDir+dataset+str(word)+'.png'
    
    #plt.plot(np.arange(1,nCategory+1),accResult[2,:],'r-s',linewidth=2.5,label='GK')
    #plt.plot(np.arange(1,nCategory+1),accResult[1,:],'b--d',linewidth=2.5,label='FCM')
    #plt.plot(np.arange(1,nCategory+1),accResult[0,:],'k-.o',linewidth=2.5,label='BoF')
    
    ind = np.arange(nCategory)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, accResult[0,:], width, color='y',label='BoF')
    rects2 = ax.bar(ind+width, accResult[2,:], width, color='k',label='GK')
    ax.legend( (rects1[0], rects2[0]), ('BoF', 'GK') )
   
    
    ax.set_xticks(ind+width)
    

    
    
    plt.xlabel('visual category',size='large')
    plt.ylabel('mAcc',size='large')
#    plt.title('%s Performance: %s ' % (title,dataset))
    plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    plt.ylim([minacc,maxacc])
    #ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(catList,rotation=30,size='medium',ha='center')
    plt.savefig(outPath,format='png')
    plt.show()
    pass

if __name__=='__main__':
    plotfuzzyresultCategory()
    pass