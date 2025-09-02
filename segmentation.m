close all;
clear all;

rng(1);

imageFolder = 'daffodilSeg/ImagesRsz256';
seglabelFolder = 'daffodilSeg/LabelsRsz256';

classNames = ["background", "flower"];
labelIDs = [0, 1];

SegClassLabel = @(filename) convertClassLabel(imread(filename));

labelds = pixelLabelDatastore(seglabelFolder, classNames, labelIDs, ...
    'ReadFcn', SegClassLabel);
imdsSeg = imageDatastore(imageFolder);

numImages = numel(imdsSeg.Files);
SegshuffledIdx = randperm(numImages);
numTrain = round(0.8 * numImages);
trainIdx = SegshuffledIdx(1:numTrain);
valIdx = SegshuffledIdx(numTrain+1:end);

imdsTrain = subset(imdsSeg, trainIdx);
imdsVal = subset(imdsSeg, valIdx);
labeldsTrain = subset(labelds, trainIdx);
labeldsVal = subset(labelds, valIdx);

trainingData = combine(imdsTrain, labeldsTrain);
validationData = combine(imdsVal, labeldsVal);

imageSize = [256 256 3];
numClasses = 2;
lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet18');

SegTrainoptions = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 10, ...
    'ValidationData', validationData, ...
    'ExecutionEnvironment','auto', ...
    'Plots', 'training-progress');

segmentnet = trainNetwork(trainingData, lgraph, SegTrainoptions);

save('segmentnet.mat', 'segmentnet');

predicted = semanticseg(imdsVal, segmentnet, 'MiniBatchSize', 4);
metrics = evaluateSemanticSegmentation(predicted, labeldsVal);

disp("Mean IoU: " + metrics.DataSetMetrics.MeanIoU);
disp("Per-class IoU:");
disp(metrics.ClassMetrics);

for i = 1:4
    img = readimage(imdsVal, i);
    label = semanticseg(img, segmentnet);
    B = labeloverlay(img, label);
    figure;
    imshow(B);
    title("DeepLab v3+ Prediction — Validation Image " + i);
end

function label = convertClassLabel(labelImage)
    % 1 = flower 
    % 0 = background 
    label = uint8(labelImage == 1);
end
