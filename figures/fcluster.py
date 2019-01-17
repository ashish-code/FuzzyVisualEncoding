'''
Created on 14 Dec 2011
fuzzy cluster the data
write the .dom file using ./dom
compute the fuzzy cluster ./cli will write the .cls file to directory
The next program will compute the fuzzy encoding from the clusters.
@author: ag00087
'''

#global imports
import os
import sys
from optparse import OptionParser
import numpy as np

#global path variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
tabDir = '/Fuzzy/FeatureUni/'
clsDir = '/Fuzzy/CLS/'
desc = 'sift'
# global variables
catidfname = 'catidlist.txt'
descext = '.'+desc
ffext = '.'+desc
tabext = '.tab'
domext = '.dom'
clsext = '.cls'
descdim = {'sift':128}

progDir = '/vol/vssp/diplecs/ash/code/IDA/'
domProgDir = progDir+'Table'+'/'
clusterProgDir = progDir+'Cluster'+'/'


parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')


def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.loadtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.loadtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def callFuzzyCluster():
    (options,args) = parser.parse_args(sys.argv[1:])
    dataset = options.dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    for categoryName in catList:
        
        pass