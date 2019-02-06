% calculate fuzzy coefficients with new imagelist

function compFuzzyCoeff(dataSet,dictSize,clustType,intDim)

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
param.dictDir = '/Dictionary/';
param.imageListDir = '/ImageList/';
param.coeffDir = '/Coeff/';
param.traindsiftDir = '/DSIFT/';
param.testdsiftDir = '/TestDSIFT/';
% read the category list in the dataset
categoryListPath = strcat(param.rootDir,param.dataSet,'/',categoryListFileName);
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
param.categoryList = categoryList;
param.nCategory = size(categoryList,1);
%
dictDataFile = strcat(param.rootDir,param.dataSet,param.dictDir,param.dataSet,num2str(param.dictSize),param.dictType,num2str(param.sampleSize),param.clustType,num2str(param.intDim),param.method,'.mat');
dict = load(dictDataFile);
param.dict = dict;

for iCategory = 1 : param.nCategory
    % load the images from imagelist
    categoryName = param.categoryList{iCategory};
    if ismember(dataSet,['Scene15','Caltech101','Caltech256'])
        coeffCatDir = strcat(param.rootDir,param.dataSet,param.coeffDir,categoryName);
        if exist(coeffCatDir,'dir') ~= 7
            mkdir(coeffCatDir)
        end
    end
    %
    [trainImageNames,~,testImageNames,~] = readImageList(categoryName,param);
    %
    nTrain = max(size(trainImageNames));
    nTest = max(size(testImageNames));
    %
    isTrain = true;
    for i = 1 : nTrain
        imageName = trainImageNames{i};
%         imageLabel = trainImageLabels(i);
        callEncoding(imageName,param,isTrain);
    end
    isTrain = false;
    for j = 1 : nTest
        imageName = testImageNames{j};
%         imageLabel = testImageLabels(j);
        callEncoding(imageName,param,isTrain);
    end
    
end
end

function callEncoding(imageName,param,isTrain)
if isTrain
    coeffFilePath = strcat(param.rootDir,param.dataSet,param.coeffDir,imageName,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.train');
else
    coeffFilePath = strcat(param.rootDir,param.dataSet,param.coeffDir,imageName,num2str(param.dictSize),param.clustType,param.method,num2str(param.intDim),'.test');
end
if exist(coeffFilePath,'file')
    return;
end

if strcmp(param.dataSet,'VOC2006')||strcmp(param.dataSet,'VOC2007')
    if isTrain
        imageFileName = strcat(param.rootDir,param.dataSet,param.traindsiftDir,imageName,'.dsift');
    else
        imageFileName = strcat(param.rootDir,param.dataSet,param.testdsiftDir,imageName,'.dsift');
    end
else
    imageFileName = strcat(param.rootDir,param.dataSet,param.traindsiftDir,imageName,'.dsift');
end


imageData = load(imageFileName);
imageData = imageData(3:130,:);
% --------------------------------------------------------------------------
% Project data to subspace
imageData = imageData';
imagesubspace = compute_mapping(imageData,param.method,param.intDim);

imgsub.X = imagesubspace;
imgsub = clust_normalize(imgsub,'range');

% if kmeans or other methods
if strcmp(param.clustType,'Kmeans')
    imagesubspace = imgsub.X;
    imagesubspace = imagesubspace';
    D = param.dict.cluster.v';
    hvqenc = dsp.VectorQuantizerEncoder('Codebook',D,'CodewordOutputPort', false,'QuantizationErrorOutputPort', false, 'OutputIndexDataType', 'int32');
    idx = step(hvqenc,imagesubspace);
    idx = idx+1;    
    coeff = zeros(1,param.dictSize);
    for i = 1 : size(imagesubspace,2)
        coeff(idx(i)) = coeff(idx(i))+1;
    end
    Favg = coeff./sum(coeff);
else
    params.m = 2;
    eval = clusteval(imgsub,param.dict,params);
    coeff = eval.f;
    
    disp(size(coeff))
    Favg = mean(coeff,1);
end
% save the coefficient to file

dlmwrite(coeffFilePath,Favg,'delimiter',',');
fprintf('%s\n',coeffFilePath);
end

function [trainImageNames,trainImageLabels,testImageNames,testImageLabels] = readImageList(categoryName,param)
    listTrainFileName = strcat(param.rootDir,param.dataSet,param.imageListDir,categoryName,'_trainval.txt');
    listTestFileName = strcat(param.rootDir,param.dataSet,param.imageListDir,categoryName,'_test.txt');
    listTrainfid = fopen(listTrainFileName,'r');
    listTrain = textscan(listTrainfid,'%s %d');
    fclose(listTrainfid);
    trainImageNames = listTrain{1};
    trainImageLabels = listTrain{2};
    listTestfid = fopen(listTestFileName,'r');
    listTest = textscan(listTestfid,'%s %d');
    fclose(listTestfid);
    testImageNames = listTest{1};
    testImageLabels = listTest{2};
    
end