close all;
clear all;

srcFolder = fullfile(pwd,"17flowers")
sortedFlowerFolder = fullfile(pwd,"newsorted17flowers")
if ~exist(sortedFlowerFolder,'dir')
mkdir(sortedFlowerFolder);
end

ImageFiles = dir(fullfile(srcFolder,'*jpg'));
for j = 1:length(ImageFiles)
filename = ImageFiles(j).name;
imageIndex = sscanf(filename,'image_%04d.jpg');
flowerclassNo = ceil(imageIndex / 80);
classFolder = fullfile(sortedFlowerFolder,['class',num2str(flowerclassNo)]);
if ~exist(classFolder,'dir')
mkdir(classFolder);
end
movefile(fullfile(srcFolder,filename),fullfile(classFolder,filename));
end
disp("flower organisation is completed");


imagereading = imageDatastore('newsorted17flowers', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
imagereading.ReadFcn = @(filename) im2double(imresize(imread(filename), [256 256]));

[imagereadingTrain, imagereadingValidation] = splitEachLabel(imagereading, 0.8, 'randomized');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-15 15], ...
    'RandXScale', [0.9 1.05], ...
    'RandYScale', [0.9 1.05], ...
    'RandXReflection', true);

augmentedTrain = augmentedImageDatastore([256 256 3], imagereadingTrain, ...
    'DataAugmentation', imageAugmenter);
augmentedValidation = augmentedImageDatastore([256 256 3], imagereadingValidation);

layers = [
    imageInputLayer([256 256 3], 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 384, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    globalAveragePooling2dLayer
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(17)
    softmaxLayer
    classificationLayer()
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 3e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augmentedValidation, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

classnet = trainNetwork(augmentedTrain, layers, options);

save('classnet.mat', 'classnet');

YPrediction = classify(classnet, augmentedValidation);
YValidation = imagereadingValidation.Labels;
valAccuracy = mean(YPrediction == YValidation);
disp("Validation Accuracy: " + valAccuracy);

YPredTrain = classify(classnet, augmentedTrain);
YTrain = imagereadingTrain.Labels;
trainAccuracy = mean(YPredTrain == YTrain);
disp("Training Accuracy: " + trainAccuracy);

figure;
confusionchart(YValidation, YPrediction);
title('Confusion Matrix - 17 Flower Classes');

confMatrix = confusionmat(YValidation, YPrediction);
classAccuracy = diag(confMatrix) ./ sum(confMatrix, 2);

figure;
bar(classAccuracy);
ylim([0 1]);
xlabel('Class Index');
ylabel('Accuracy');
title('Per-Class Accuracy - 17 Flower Classes');

%% Misclassified Flowers
idx = find(YPrediction ~= YValidation);
numMisclassDisplay = min(16, numel(idx));

figure;
for i = 1:numMisclassDisplay
    subplot(4, 4, i);
    imshow(readimage(imagereadingValidation, idx(i)));
    title("True: " + string(YValidation(idx(i))) + ...
          ", Pred: " + string(YPrediction(idx(i))));
end
sgtitle("Misclassified Images");





