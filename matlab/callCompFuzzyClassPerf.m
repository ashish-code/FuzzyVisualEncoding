% call compFuzzyClassPerf()

function callCompFuzzyClassPerf(dataSet,clustType)
dictSizes = [25,50,100,200,400];
intDims = [2,3,12,128];
ndictSizes = max(size(dictSizes));
nintDims = max(size(intDims));

for i = 1 : ndictSizes
    dictSize = dictSizes(i);
    for j = 1 : nintDims
        intDim = intDims(j);
        fprintf('%d,%d\n',dictSize,intDim);
        try
            compFuzzyClassPerf(dataSet,dictSize,clustType,intDim);
        catch err
            fprintf('%s\n',err.identifier);
        end
    end
end
end