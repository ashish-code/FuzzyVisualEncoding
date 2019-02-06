% plot the FCM dictionary for real world data
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
dsiftDir = '/DSIFT/';
intDim = 2;
dataSet = 'VOC2006';
dictSize = 25;
dictType = 'universal';
sampleSize = 100000;
method = 'Kmeans';
subspaceMethod = 'PCA';

%parameters
param.c=4;
param.m=2;
param.e=1e-12;

imageName = '000017';

% load image data
imageFilePath = [(rootDir),(dataSet),(dsiftDir),(imageName),'.dsift'];
imageData = load(imageFilePath);
imageData = imageData(3:130,:);
% --------------------------------------------------------------------------
imagesubspace = compute_mapping(imageData',subspaceMethod,intDim);

% load dictionary
dictDataFile = [(rootDir),(dataSet),(dictDir),(dataSet),num2str(dictSize),(dictType),num2str(sampleSize),(method),num2str(intDim),(subspaceMethod),'.mat'];
dict = load(dictDataFile);

imgsub.X = imagesubspace;
imgsub = clust_normalize(imgsub,'range');
param.m = 2;
eval = clusteval(imgsub,dict,param);

plot(imgsub.X(:,1),imgsub.X(:,2),'kd','MarkerSize',4,'MarkerFaceColor','k','MarkerEdgeColor','k');
hold on
plot(dict.cluster.v(:,1),dict.cluster.v(:,2),'rs','MarkerSize',6,'MarkerFaceColor','r','MarkerEdgeColor','r');
if(method == 'Kmeans')
    voronoi(dict.cluster.v(:,1),dict.cluster.v(:,2),'b-');
end
xlabel('x');
ylabel('y');
title('Fuzzy K-means Fuzzy Partition | VOC-2006');
% hold on
%draw contour-map
%new.X=imgsub.X;
%eval=clusteval(new,dict,param);