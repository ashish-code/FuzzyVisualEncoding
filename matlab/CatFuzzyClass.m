function CatFuzzyClass(dataSet,dictSize,clustMethod)
dictType = 'balanced';
intDim = 3;
subspaceMethod = 'PCA';
rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
sampleDir = '/collated/';
dictDir = '/Dictionary/';
mappingDir = '/Mapping/';
imageListDir = '/ImageLists/';
coeffDir = '/Coeff/';
coeffPerfDir = '/CoeffPerf/';
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
% Compute category-specific dictionary
for iCategory = 1 : nCategory
    dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),clustMethod,num2str(intDim),(subspaceMethod),'.mat'];
    if exist(dictDataFile,'file')
        continue;
    end
    sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.bal'];        
    mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),subspaceMethod,num2str(intDim),'.mat'];
    fprintf('%s\n',sampleDataFile);
    callFuzzy(sampleDataFile,dictDataFile,dictSize,clustMethod,mappingFile);
    fprintf('%s\n',dictDataFile)
end

% Compute Coefficients

listSize = 30;
for iCategory = 1 : nCategory
    dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),clustMethod,num2str(intDim),(subspaceMethod),'.mat'];
    mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),subspaceMethod,num2str(intDim),'.mat'];
    dict = load(dictDataFile);
    mapping = load(mappingFile);
    % load the images from imagelist    
    if ismember(dataSet,['Scene15','Caltech101','Caltech256'])
        coeffCatDir = [(rootDir),(dataSet),(coeffDir),categoryList{iCategory}];
        if exist(coeffCatDir,'dir') ~= 7
            mkdir(coeffCatDir)
        end
    end
    %
    
    listTrainPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSize),'.pos'];
    listValPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSize),'.pos'];
    listTrainNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSize),'.neg'];
    listValNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSize),'.neg'];
    %        
    fid = fopen(listTrainPosFile,'r');
    listTrainPos = textscan(fid,'%s');
    fclose(fid);
    listTrainPos = listTrainPos{1};
    %
    fid = fopen(listValPosFile,'r');
    listValPos = textscan(fid,'%s');
    fclose(fid);
    listValPos = listValPos{1};
    %
    fid = fopen(listTrainNegFile,'r');
    listTrainNeg = textscan(fid,'%s');
    fclose(fid);
    listTrainNeg = listTrainNeg{1};
    %
    fid = fopen(listValNegFile,'r');
    listValNeg = textscan(fid,'%s');
    fclose(fid);
    listValNeg = listValNeg{1};
    %
    nListTrainPos = size(listTrainPos,1);
    nListValPos = size(listValPos,1);
    nListTrainNeg = size(listTrainNeg,1);
    nListValNeg = size(listValNeg,1);     

    % Train ; Pos
    for iter = 1 : nListTrainPos
        imageName = listTrainPos{iter};
        callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,clustMethod,subspaceMethod,mapping);            
    end

    % Val ; Pos
    for iter = 1 : nListValPos
        imageName = listValPos{iter};
        callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,clustMethod,subspaceMethod,mapping);           
    end

    % Train ; Neg
    for iter = 1 : nListTrainNeg
        imageName = listTrainNeg{iter};
        callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,clustMethod,subspaceMethod,mapping);           
    end

    % Val ; Neg
    for iter = 1 : nListValNeg
        imageName = listValNeg{iter};
        callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,clustMethod,subspaceMethod,mapping);         
    end
    
end

% Compute classification performance
coeffPerfFileAvg  = strcat(rootDir,dataSet,coeffPerfDir,clustMethod,num2str(dictSize),subspaceMethod,num2str(intDim),'.bal');
coeffPerfAvg = zeros(nCategory,3);
for iCategory = 1 : nCategory
    category = categoryList{iCategory};
    fprintf('%s\n',category);
    listTrainPosFile = [(rootDir),(dataSet),(imageListDir),(category),'Train',num2str(listSize),'.pos'];
    listValPosFile = [(rootDir),(dataSet),(imageListDir),(category),'Val',num2str(listSize),'.pos'];
    listTrainNegFile = [(rootDir),(dataSet),(imageListDir),(category),'Train',num2str(listSize),'.neg'];
    listValNegFile = [(rootDir),(dataSet),(imageListDir),(category),'Val',num2str(listSize),'.neg'];
            
    fid = fopen(listTrainPosFile,'r');listTrainPos = textscan(fid,'%s');fclose(fid);
    listTrainPos = listTrainPos{1};

    fid = fopen(listValPosFile,'r');listValPos = textscan(fid,'%s');fclose(fid);
    listValPos = listValPos{1};

    fid = fopen(listTrainNegFile,'r');listTrainNeg = textscan(fid,'%s');fclose(fid);
    listTrainNeg = listTrainNeg{1};

    fid = fopen(listValNegFile,'r');listValNeg = textscan(fid,'%s');fclose(fid);
    listValNeg = listValNeg{1};

    nListTrainPos = size(listTrainPos,1);
    nListValPos = size(listValPos,1);
    nListTrainNeg = size(listTrainNeg,1);
    nListValNeg = size(listValNeg,1);

    FTrainPosAvg = ones(nListTrainPos,dictSize+1);
    FTrainNegAvg = zeros(nListTrainNeg,dictSize+1);
    FValPosAvg = ones(nListValPos,dictSize+1);
    FValNegAvg = zeros(nListValNeg,dictSize+1);        

    % Train ; Pos
    for iter = 1 : nListTrainPos
        imageName = listTrainPos{iter};
        coeffFilePathAvg = strcat(rootDir,dataSet,coeffDir,imageName,num2str(dictSize),dictType,num2str(sampleSize),clustMethod,subspaceMethod,'.avg');
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end                 
        FTrainPosAvg(iter,1:dictSize) = Favg;            
    end

    % Val ; Pos
    for iter = 1 : nListValPos
        imageName = listValPos{iter};
        coeffFilePathAvg = strcat(rootDir,dataSet,coeffDir,imageName,num2str(dictSize),dictType,num2str(sampleSize),clustMethod,subspaceMethod,'.avg');
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier)
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end            
        FValPosAvg(iter,1:dictSize) = Favg;           
    end

    % Train ; Neg
    for iter = 1 : nListTrainNeg
        imageName = listTrainNeg{iter};
        coeffFilePathAvg = strcat(rootDir,dataSet,coeffDir,imageName,num2str(dictSize),dictType,num2str(sampleSize),clustMethod,subspaceMethod,'.avg');
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end           
        FTrainNegAvg(iter,1:dictSize) = Favg;            
    end

    % Val ; Neg
    for iter = 1 : nListValNeg
        imageName = listValNeg{iter};
        coeffFilePathAvg = strcat(rootDir,dataSet,coeffDir,imageName,num2str(dictSize),dictType,num2str(sampleSize),clustMethod,subspaceMethod,'.avg');
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end            
        FValNegAvg(iter,1:dictSize) = Favg;          
    end

FTrainAvg = [FTrainPosAvg;FTrainNegAvg];
FValAvg = [FValPosAvg;FValNegAvg];

%---------------------------------------------------------------------
% CLASSIFICATION
%---------------------------------------------------------------------
% echo pipeline stage: classification
fprintf('%s\n','classification');
cdir = pwd;
cd '/vol/vssp/diplecs/ash/code/libsvm-3.11/matlab/';
% train the classifier
svmStruct = svmtrain(FTrainAvg(:,dictSize+1),FTrainAvg(:,1:dictSize), '-h 0');
% predict label of data using the trained classifier
[predLabel, ~, predValues] = svmpredict(FValAvg(:,dictSize+1), FValAvg(:,1:dictSize), svmStruct);
cd (cdir);
[~, ~, ~, averagePrecision] = precisionRecall(predValues, FValAvg(:,dictSize+1), 'r');
[~, ~, AUC] = ROCcurve(predValues, FValAvg(:,dictSize+1));
cp = classperf(FValAvg(:,dictSize+1),predLabel);
fprintf('%s\t%f\t%f\t%f\n',categoryList{iCategory},averagePrecision,AUC,cp.CorrectRate);
coeffPerfAvg(iCategory,:) = [averagePrecision,AUC,cp.CorrectRate];
end
dlmwrite(coeffPerfFileAvg,coeffPerfAvg,'delimiter',',');
fprintf('%s\n',coeffPerfFileAvg);

end

function callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,clustMethod,subspaceMethod,mapping)
rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
dsiftDir = '/DSIFT/';

coeffFilePathAvg = strcat(rootDir,dataSet,coeffDir,imageName,num2str(dictSize),dictType,num2str(sampleSize),clustMethod,subspaceMethod,'.avg');


imageFilePath = [(rootDir),(dataSet),(dsiftDir),(imageName),'.dsift'];
imageData = load(imageFilePath);
imageData = imageData(3:130,:);
imageData = imageData';
% --------------------------------------------------------------------------
imagesubspace = out_of_sample(imageData,mapping); % mapping sample data down

% if kmeans or other methods
if strcmp(clustMethod,'Kmeans')
    imgsub.X = imagesubspace;
    imgsub = clust_normalize(imgsub,'range');
    imagesubspace = imgsub.X;
    imagesubspace = imagesubspace';
    D = dict.cluster.v';
    hvqenc = dsp.VectorQuantizerEncoder('Codebook',D,'CodewordOutputPort', false,'QuantizationErrorOutputPort', false, 'OutputIndexDataType', 'uint16');
    idx = step(hvqenc,imagesubspace);
    idx = idx+1;    
    coeff = zeros(1,dictSize);
    for i = 1 : size(imagesubspace,2)
        coeff(idx(i)) = coeff(idx(i))+1;
    end
    Favg = coeff./sum(coeff);
else
    imgsub.X = imagesubspace;
    imgsub = clust_normalize(imgsub,'range');
    param.m = 2;
    eval = clusteval(imgsub,dict,param);
    coeff = eval.f;
    Favg = mean(coeff,1);    
end
% --------------------------------------------------------------------------
dlmwrite(coeffFilePathAvg,Favg,'delimiter',',');    
fprintf('%s\n',coeffFilePathAvg);
end

function callFuzzy(sampleDataFile,dictDataFile,dictSize,clustMethod,mappingFile)
progDir = '/vol/vssp/diplecs/ash/code/FuzzyClusteringToolbox_m/FUZZCLUST/';    
sampleData = load(sampleDataFile);
mapping = load(mappingFile);
sampleData = sampleData';
nVec = size(sampleData,1);
nSample = 50000;
rndSample = randsample(nVec,nSample);
sampled = sampleData(rndSample,:);
sample = out_of_sample(sampled,mapping); % mapping sample data down
% remove Nan and Inf matrix elements
sample(isnan(sample))=0;
sample(isinf(sample))=0;    
data.X = sample;               
clear sampleData;    
cd (progDir);    
%data normalization
data = clust_normalize(data,'range');
param.c = dictSize;
param.vis = 0;
if (strcmp(clustMethod,'Kmeans'))
    result = Kmeans(data,param);
elseif (strcmp(clustMethod,'FCM'))
    param.m=2;
    param.iter = 1000;
    param.e=1e-12;
    param.ro=ones(1,param.c);
    param.val=1;
    result = FCMclust(data,param);
elseif(strcmp(clustMethod,'GK'))
    param.m=2;
    param.iter = 1000;
    param.e=1e-12;
    param.ro=ones(1,param.c);
    param.val=3;
    result = GKclust(data,param);
elseif (strcmp(clustMethod,'GG'))
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
end

