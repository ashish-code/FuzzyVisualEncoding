'''
Created on 14 Dec 2011
Write the feature vector in tab format for clustering
@author: ag00087
'''

#global imports
import sys
from optparse import OptionParser
import numpy as np
import os


#global path variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
tabDir = '/Fuzzy/Feature/'
desc = 'sift'
# global variables
catidfname = 'catidlist.txt'
descext = '.'+desc
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

def writeTabFeatureVector():
    (options,_) = parser.parse_args(sys.argv[1:])
    dataset = options.dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dim = descdim.get(desc)
    
    for catName in catList:
        catFilePath = rootDir+dataset+dataDir+catName+descext
        print 'reading %s'%(catName)
        catData = np.loadtxt(catFilePath,dtype=np.int16,usecols=np.arange(2,dim+2)) # this would not hold in a generic scenario
        catTabFileDir = rootDir+dataset+tabDir
        if not os.path.exists(catTabFileDir):
            os.mkdir(catTabFileDir)
        catTabFilePath = catTabFileDir+catName+tabext
        print 'writing %s'%(catName)
        np.savetxt(catTabFilePath, catData, fmt='%d', delimiter='\t')
        pass

if __name__=='__main__':
    writeTabFeatureVector()
    pass