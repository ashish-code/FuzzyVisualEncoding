'''
Created on 13 Dec 2011
Write the feature descriptors for each category in ../FeatureMatrix/..
in a .tab file. The descriptor should have equal number of feature 
vectors from other categories as negative data set. For encoding to 
work the negative images should be rather than feature descriptors. 
@author: ag00087
'''

#global imports
import sys
from optparse import OptionParser
import numpy as np


#global path variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
tabDir = '/Fuzzy/FeatureUni/'
desc = 'sift'
# global variables
catidfname = 'catidlist.txt'
descext = '.'+desc
ffext = '.'+desc
tabext = '.tab'
descdim = {'sift':128}

parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')
parser.add_option('-s','--nclusterSample',type='int',metavar='clusterSampleSize',dest='nClusterSample',default=10000,help='number of samples for clustering')

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.loadtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.loadtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def writeFuzzyFeatureVectorBinaryClassifier():
    (options,_) = parser.parse_args(sys.argv[1:])
    dataset = options.dataset
    nClusterSample = options.nClusterSample
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dim = descdim.get(desc)
    nCategory = len(catList)
    nSamplePerCategory = int(np.round(nClusterSample/nCategory))
    dataPath = rootDir+dataset+dataDir
    clusterData = None
    for catName in (catList):
        catFilePath = dataPath+catName+descext
        print 'reading %s'%(catName)
        catData = np.loadtxt(catFilePath,dtype=np.int16,usecols=np.arange(2,dim+2)) # this would not hold in a generic scenario
        if(catData.shape[0] <= nSamplePerCategory):
            catSample = catData
        else:
            rndsample = np.random.randint(0,catData.shape[0],nSamplePerCategory)
            catSample = catData[rndsample,:]
        if(clusterData == None):
            clusterData = catSample
        else:
            clusterData = np.concatenate((clusterData,catSample),axis=0)
    
    outDir = rootDir+dataset+tabDir
    outPath = outDir+dataset+tabext
    print 'saving...'
    np.savetxt(outPath, clusterData, fmt='%d', delimiter='\t')
    
    
if __name__=='__main__':
    writeFuzzyFeatureVectorBinaryClassifier()
    pass
