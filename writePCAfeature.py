'''
Created on 15 Dec 2011
PCA down to 10 dimensions based on collated data.
pcaModel = pca.fit(X)
pcaProj = pcaModel.transform(Y)
@author: a.gupta@surrey.ac.uk
'''

#global imports
import sys
from optparse import OptionParser
import numpy as np
from sklearn.decomposition import PCA


#global path variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
tabDir = '/Fuzzy/Feature/'
fvectorDir = '/Fuzzy/FeatureUni/'
desc = 'sift'
# global variables
catidfname = 'catidlist.txt'
descext = '.'+desc
tabext = '.tab'
descdim = {'sift':128}


parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')
parser.add_option('-l','--lowerDim',type='int',metavar='lowerDim',dest='lowerDim',default=10,help='lower dimension to project sift data to must be 1 <= lowerDim <= 128')

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.loadtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.loadtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def writeLowerdim():
    (options,_) = parser.parse_args(sys.argv[1:])
    dataset = options.dataset

    lowerdim = 10
    fvectorPath = rootDir+dataset+fvectorDir+dataset+tabext
    fvector = np.loadtxt(fvectorPath, dtype=np.int,delimiter='\t')
    pca = PCA(n_components = lowerdim)
    pcaModel = pca.fit(fvector)
    fv10 = pcaModel.transform(fvector)
    fv10Path = rootDir+dataset+fvectorDir+dataset+str(lowerdim)+tabext
    print 'writing %s'%(dataset)
    np.savetxt(fv10Path, fv10, fmt='%f', delimiter='\t')
    
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    
    for catName in catList:
        catTabFileDir = rootDir+dataset+tabDir
        catTabFilePath = catTabFileDir+catName+tabext
        print 'reading %s...'%(catName)
        catData = np.loadtxt(catTabFilePath, dtype=np.int, delimiter='\t')
        catData10 = pcaModel.transform(catData)
        catTab10FilePath = catTabFileDir+catName+str(lowerdim)+tabext
        catData10 = catData10[:100,:]
        np.savetxt(catTab10FilePath,catData10,fmt='%f',delimiter='\t')
        pass
    
if __name__=='__main__':
    writeLowerdim()
    pass