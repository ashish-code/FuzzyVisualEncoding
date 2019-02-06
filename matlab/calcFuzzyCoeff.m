% calc fuzzy clustering coeff matrices
function calcFuzzyCoeff(dataSet,dictType,dictSize,sampleSize,method,subspaceMethod)

% initialize matlab
cdir = pwd;
cd ~;
startup;
cd (cdir);
%
rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
dictDir = '/Dictionary/';
imageListDir = '/ImageLists/';
coeffDir = '/Coeff/';
% read the category list in the dataset
categoryListPath = [(rootDir),(dataSet),'/',(categoryListFileName)];
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
nCategory = size(categoryList,1);
listSizes = 30;
nListSizes = max(size(listSizes));
intDim = 3;
dictDataFile = [(rootDir),(dataSet),(dictDir),(dataSet),num2str(dictSize),(dictType),num2str(sampleSize),(method),num2str(intDim),(subspaceMethod),'.mat'];
dict = load(dictDataFile);
for iCategory = 1 : nCategory
    % load the images from imagelist    
    if ismember(dataSet,['Scene15','Caltech101','Caltech256'])
        coeffCatDir = [(rootDir),(dataSet),(coeffDir),categoryList{iCategory}];
        if exist(coeffCatDir,'dir') ~= 7
            mkdir(coeffCatDir)
        end
    end
    %
    for iListSize = 1 : nListSizes
        listTrainPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSizes(iListSize)),'.pos'];
        listValPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSizes(iListSize)),'.pos'];
        listTrainNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSizes(iListSize)),'.neg'];
        listValNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSizes(iListSize)),'.neg'];
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
            callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,method,subspaceMethod);            
        end
        
        % Val ; Pos
        for iter = 1 : nListValPos
            imageName = listValPos{iter};
            callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,method,subspaceMethod);           
        end
        
        % Train ; Neg
        for iter = 1 : nListTrainNeg
            imageName = listTrainNeg{iter};
            callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,method,subspaceMethod);           
        end
        
        % Val ; Neg
        for iter = 1 : nListValNeg
            imageName = listValNeg{iter};
            callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,method,subspaceMethod);         
        end
    end
end
end

function callEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,method,subspaceMethod)
rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
dsiftDir = '/DSIFT/';
algo = 'dl';
param = 'neg';
intDim = 3;

coeffFilePathAvg = strcat(rootDir,dataSet,coeffDir,imageName,num2str(dictSize),dictType,num2str(sampleSize),algo,num2str(param),method,subspaceMethod,'.avg');
% if exist(coeffFilePathAvg,'file')
%     return;
% end

imageFilePath = [(rootDir),(dataSet),(dsiftDir),(imageName),'.dsift'];
imageData = load(imageFilePath);
imageData = imageData(3:130,:);

% --------------------------------------------------------------------------
% Project data to subspace
imagesubspace = compute_mapping(imageData',subspaceMethod,intDim);


% if kmeans or other methods
if strcmp(method,'Kmeans')
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
    param.m = 2;
    eval = clusteval(imgsub,dict,param);
    coeff = eval.f;
    Favg = mean(coeff,2);
end
% --------------------------------------------------------------------------
dlmwrite(coeffFilePathAvg,Favg,'delimiter',',');    
fprintf('%s\n',coeffFilePathAvg);
end