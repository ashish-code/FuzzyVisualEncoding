function graphFuzzyCoeffPerf()
dataSets = {'VOC2006','VOC2007','VOC2010','Scene15'};
nDataSet = max(size(dataSets));
rootDir = '/vol/vssp/diplecs/ash/Data/';
% initialize matlab
cdir = pwd;
cd ~;
startup;
cd (cdir);
%
subspaceMethod = 'PCA';
clustMethods = {'Kmeans';'FCM';'GK';'GG'};

resultDir = 'Result/';
coeffPerfDir = '/CoeffPerf/';

nMethods = max(size(clustMethods));



for iDataSet = 1 : nDataSet
    dataSet = dataSets{iDataSet};
    resultFileName = strcat(rootDir,resultDir,dataSet,'fuzzycoeffperf.csv');
    resultFile = fopen(resultFileName,'w');
    fprintf(resultFile,'%s,%s,%s,%s\n', 'Category', 'Method', 'mAP', 'AUC');
    categoryListFileName = 'categoryList.txt';
    categoryListPath = strcat(rootDir,dataSet,'/',categoryListFileName);
    fid = fopen(categoryListPath);
    categoryList = textscan(fid,'%s');
    categoryList = categoryList{1};
    fclose(fid);
    nCategory = size(categoryList,1);
    for iMethod = 1 : nMethods
        method = clustMethods{iMethod};
        coeffPerfFileAvg  = strcat(rootDir,dataSet,coeffPerfDir,algo,num2str(param),method,subspaceMethod,'.avg');
        coeffPerfAvg = dlmread(coeffPerfFileAvg,',');
        for iCategory = 1 : nCategory
            category = categoryList{iCategory};
            map = coeffPerfAvg(iCategory,1);
            auc = coeffPerfAvg(iCategory,2);
            fprintf(resultFile,'%s,%s,%s,%f,%f\n',dataSet,category,method,map,auc);
            fprintf('%s,%s,%s,%f,%f\n',dataSet,category,method,map,auc);
        end
    end
end
fclose(resultFile);
end