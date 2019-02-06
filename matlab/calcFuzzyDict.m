% calcFuzzyDict
function calcFuzzyDict(dataSet,dictSize,dictType,clustType,intDim,method)
% function calcFuzzyDict(dataSet,dictType,dictSize,sampleSize,clustType)
% dataSet : the data set utilized
% dictType: categorical; universal; balanced
% dictSize: number of elements in the dictionary
% sampleSize: the number of random samples used to compute the dictionary

rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
sampleDir = '/collated/';
dictDir = '/Dictionary/';
mappingDir = '/Mapping/';

sampleSize = 100000;

% initialize matlab
cdir = pwd;
cd ~
startup;
cd (cdir)

% read the category list in the dataset
categoryListPath = [(rootDir),(dataSet),'/',(categoryListFileName)];
fid = fopen(categoryListPath);
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
nCategory = size(categoryList,1);
if strcmp(dictType,'universal')
    sampleDataFile = [(rootDir),(dataSet),(sampleDir),(dataSet),num2str(sampleSize),'.uni'];
    dictDataFile = [(rootDir),(dataSet),(dictDir),(dataSet),num2str(dictSize),(dictType),num2str(sampleSize),clustType,num2str(intDim),(method),'.mat'];
    mappingFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),method,num2str(intDim),'.mat'];
    fprintf('%s\n',sampleDataFile);
    % load the sample data file
    callFuzzy(sampleDataFile,dictDataFile,dictSize,clustType,mappingFile);
    fprintf('%s\n',sampleDataFile)
    
elseif strcmp(dictType,'categorical')
    for iCategory = 1 : nCategory
        sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.cat'];
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),clustType,num2str(intDim),(method),'.mat'];        
        mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),method,num2str(intDim),'.mat'];
        fprintf('%s\n',sampleDataFile);
        callFuzzy(sampleDataFile,dictDataFile,dictSize,clustType,mappingFile);
        fprintf('%s\n',dictDataFile)
    end
elseif strcmp(dictType,'balanced')
    for iCategory = 1 : nCategory
        sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.bal'];
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),clustType,num2str(intDim),(method),'.mat'];        
        mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),method,num2str(intDim),'.mat'];
        fprintf('%s\n',sampleDataFile);
        callFuzzy(sampleDataFile,dictDataFile,dictSize,clustType,mappingFile);
        fprintf('%s\n',dictDataFile)
    end
end

end

function callFuzzy(sampleDataFile,dictDataFile,dictSize,clustType,mappingFile)
    progDir = '/vol/vssp/diplecs/ash/code/FuzzyClusteringToolbox_m/FUZZCLUST/';
%     if exist(dictDataFile,'file')
%        return;
%     end
    
    sampleData = load(sampleDataFile);
    mapping = load(mappingFile);
    sampleData = sampleData';
    nVec = size(sampleData,1);
    nSample = 50000;
    rndSample = randsample(nVec,nSample);
    sampled = sampleData(rndSample,:);
    sample = out_of_sample(sampled,mapping);
    % remove Nan and Inf matrix elements
    sample(isnan(sample))=0;
    sample(isinf(sample))=0;
    
    % ----------------------------------------------------------------
    % Project to 3 dimensional sub-space using PCA
    % use the learnt projection matrix; the significance to a learnt projection matrix will be found
    % during image descriptor encoding
    
    % ----------------------------------------------------------------
    data.X = sample;               % transpose of data matrix
    clear sampleData;
    
    cd (progDir);
    
    %data normalization
    data = clust_normalize(data,'range');
    
    param.c = dictSize;
    param.vis = 0;
    if (strcmp(clustType,'Kmeans'))
        result = Kmeans(data,param);
    elseif (strcmp(clustType,'FCM'))
        param.m=2;
        param.iter = 1000;
        param.e=1e-12;
        param.ro=ones(1,param.c);
        param.val=1;
        result = FCMclust(data,param);
    elseif(strcmp(clustType,'GK'))
        param.m=2;
        param.iter = 1000;
        param.e=1e-12;
        param.ro=ones(1,param.c);
        param.val=3;
        result = GKclust(data,param);
    elseif (strcmp(clustType,'GG'))
        param.m=2;
        param.iter = 1000;
        param.e=1e-12;
        param.val = 1;
        result = FCMclust(data,param);
        param.c=result.data.f;
        result = GGclust(data,param);
    else
        disp('unrecognized clustering method');
        quit;
    end
    save(dictDataFile,'-struct','result');
    disp(dictDataFile);
end