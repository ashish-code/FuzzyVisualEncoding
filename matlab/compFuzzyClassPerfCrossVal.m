% compute fuzzy classification performance
function compFuzzyClassPerfCrossVal(dataSet,dictSize,clustType,intDim)

% legacy default parameters
param.dictType = 'universal';
param.sampleSize = 100000;
param.method = 'PCA';
%
param.dictSize = dictSize;
param.clustType = clustType;
param.intDim = intDim;
param.dataSet = dataSet;
% initialize matlab
cdir = pwd;
cd ~;
startup;
cd (cdir);
%
param.rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
param.imageListDir = '/ImageList/';
param.coeffDir = '/Coeff/';
param.coeffPerfDir = '/CoeffPerf/';
% read the category list in the dataset
categoryListPath = strcat(param.rootDir,param.dataSet,'/',categoryListFileName);
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
param.categoryList = categoryList;
param.nCategory = size(categoryList,1);
%
classPerfFileName = strcat(param.rootDir,param.dataSet,param.coeffPerfDir,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.cvp');
classPerffid = fopen(classPerfFileName,'w');
for iCategory = 1 : param.nCategory
    categoryName = param.categoryList{iCategory};
    [trainImageNames,trainImageLabels,~,~] = readImageList(categoryName,param);
    %
    nTrain = max(size(trainImageNames));
%     nTest = max(size(testImageNames));
    %
    coeffTrain = zeros(nTrain,param.dictSize);
%     coeffTest = zeros(nTest,param.dictSize);
    %
    isTrain = true;
    for i = 1 : nTrain
        imageName = trainImageNames{i};
        if isTrain==true
            coeffFilePath = strcat(param.rootDir,param.dataSet,param.coeffDir,imageName,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.train');
        else
            coeffFilePath = strcat(param.rootDir,param.dataSet,param.coeffDir,imageName,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.test');
        end
        coeff = dlmread(coeffFilePath,',');
        
        coeffTrain(i,:) = coeff;
    end
%     isTrain = false;
%     for i = 1 : nTest
%         imageName = testImageNames{i};
%         if isTrain==true
%             coeffFilePath = strcat(param.rootDir,param.dataSet,param.coeffDir,imageName,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.train');
%         else
%             coeffFilePath = strcat(param.rootDir,param.dataSet,param.coeffDir,imageName,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.test');
%         end
%         coeff = dlmread(coeffFilePath,',');
%         coeffTest(i,:) = coeff;
%     end
    
    if strcmp(param.dataSet,'Caltech101')||strcmp(param.dataSet,'Scene15')||strcmp(param.dataSet,'VOC2006')||strcmp(param.dataSet,'VOC2007')
        idxtrainpos = find(trainImageLabels==1);
        idxtrainneg = find(trainImageLabels~=1);
        idxtrainneg = idxtrainneg(1:max(size(idxtrainpos)));
        idxtrain = [idxtrainpos;idxtrainneg];
%         idxtestpos = find(testImageLabels==1);
%         idxtestneg = find(testImageLabels~=1);
%         idxtestneg = idxtestneg(1:max(size(idxtestpos)));
%         idxtest = [idxtestpos;idxtestneg];
        %
        trainImageLabels = trainImageLabels(idxtrain);
%         testImageLabels = testImageLabels(idxtest);
        coeffTrain = coeffTrain(idxtrain,:);
%         coeffTest = coeffTest(idxtest,:);        
    end
    %
    %---------------------------------------------------------------------
    % CLASSIFICATION
    %---------------------------------------------------------------------
    % echo pipeline stage: classification
    fprintf('%s\n','classification');
    cdir = pwd;
    cd '/vol/vssp/diplecs/ash/code/libsvm-3.11/matlab/';
    %
    % change the train and test labels 1 <- 1 and 0 <- -1
    trainImageLabels(trainImageLabels==-1)=0;
%     testImageLabels(testImageLabels==-1)=0;
    
    % setup cross validation
    nLabels = max(size(trainImageLabels));
    nFold = 10;
    ap = zeros(1,nFold);
    auc = zeros(1,nFold);
    f1 = zeros(1,nFold);
    for cvidx = 1 : nFold
    [TrainIdx, TestIdx] = crossvalind('HoldOut', nLabels, 0.5);
    % train the classifier     
    svmStruct = svmtrain(trainImageLabels(TrainIdx),coeffTrain(TrainIdx,:), '-s 0 -b 0 -t 2 -h 1 -c 100 -d 10');            
    % predict label of data using the trained classifier
    [~, ~, predValues] = svmpredict(trainImageLabels(TestIdx), coeffTrain(TestIdx), svmStruct, '-b 0');
    cd (cdir);    
    [recall, precision, ~, averagePrecision] = precisionRecall(predValues, trainImageLabels(TestIdx));
    f1 = 2*((precision.*recall)/(precision+recall));
    [~, ~, AUC] = ROCcurve(predValues, trainImageLabels(TestIdx));
    ap(cvidx) = averagePrecision;
    auc(cvidx) = AUC;
    f1(cvidx) = f1;
    end
    map = nanmean(ap);
    mauc = nanmean(auc);
    mf1 = nanmean(f1);
    
    fprintf('%f\t%f\t%f\n',map,mauc,mf1);
    fprintf(classPerffid,'%s,%f,%f,%f\n',categoryName,map,mauc,mf1);     
end
fclose(classPerffid);
fprintf('%s\n',classPerfFileName);
end

function [trainImageNames,trainImageLabels,testImageNames,testImageLabels] = readImageList(categoryName,param)
    listTrainFileName = strcat(param.rootDir,param.dataSet,param.imageListDir,categoryName,'_trainval.txt');
    listTestFileName = strcat(param.rootDir,param.dataSet,param.imageListDir,categoryName,'_test.txt');
    listTrainfid = fopen(listTrainFileName,'r');
    listTrain = textscan(listTrainfid,'%s %d');
    fclose(listTrainfid);
    trainImageNames = listTrain{1};
    trainImageLabels = double(listTrain{2});
    listTestfid = fopen(listTestFileName,'r');
    listTest = textscan(listTestfid,'%s %d');
    fclose(listTestfid);
    testImageNames = listTest{1};
    testImageLabels = double(listTest{2});    
end