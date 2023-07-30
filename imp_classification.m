clear
close all
clc
%%
file = dir();
i=1;
ii=0;
%% Read Images:
 for la=11:15
     
     folder_name = sprintf('%s\\%s\\*.jpg',file(la).folder,file(la).name);
     file1=dir(folder_name);
     for fl_img=1:length(file1)
         close all
         img_name=sprintf('%s\\%s',file1(fl_img).folder,file1(fl_img).name);
         im = imread(img_name);
         if size(im,3)==3
             im=rgb2gray(im);
         end
         data_x(:,:,:,i)=imresize(im,[480 480]);
         data_y(i)=categorical(ii);
         i=i+1
         
%          figure, imshow(im);
     end
    ii=ii+1; 
end
%%
a=data_x;
h=size(a(:,:,:,1),1);
w=size(a(:,:,:,1),2);
test_x = zeros([h w  3]);

for i=1:size(a,4)
test_x=a(:,:,:,i);   
    
 % initialize cnn
 cnn.start=1; % just intiationg cnn object
 cnn=initcnn(cnn,[h w]);
%load trained cnn network
 %following commented code shows the cnn we have trained and saved in above
%  %file.
 cnn=cnnAddConvLayer(cnn, 9, [5 5], 'rect');
 cnn=cnnAddPoolLayer(cnn, 2, 'mean');
 cnn=cnnAddConvLayer(cnn, 15, [5 5], 'rect');
 cnn=cnnAddPoolLayer(cnn, 2, 'mean');
 cnn=cnnAddConvLayer(cnn, 21, [5 5], 'rect');
 cnn=cnnAddPoolLayer(cnn, 2, 'mean');
% 
cnn = ffcnn(cnn, test_x);
out = cell2mat(cnn.layers{1, 7}.featuremaps );
 feat = out(:).';
train(i,:)=feat;
train1(i,:) = {out};
i
end

clear test_x
%%
 
for i=1:size(train,1) 
    x=train(i,:);
    data(i,:)=[x grp2idx(data_y(i))];
end

[m,n] = size(data);
P = 0.70 ;
idx = randperm(m);
setTrain = data(idx(1:round(P*m)),:); 
setTest = data(idx(round(P*m)+1:end),:);
inputSize=480;


for i=1:size(setTrain,1)
    train_x{i,1}=imresize(setTrain(i,end-1),[inputSize inputSize]);
    train_y(i,:)=categorical(setTrain(i,end));
end

for i=1:size(setTest,1)
    test_x{i,1}=imresize(setTest(i,end-1),[inputSize inputSize]);
    test_y(i,:)=categorical(setTest(i,end));
end

%%
maxEpochs = 150;
miniBatchSize = 30;

numFeatures = 480;
numHiddenUnits1 = 125;
numHiddenUnits2 = 100;
numClasses = 5;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits2,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ... 
    'Verbose',0, ...
    'Plots','training-progress');


net1 = trainNetwork(train_x,train_y,layers,options);

YPred1 = classify(net1,test_x);

%%

figure()
plotconfusion(test_y,YPred1)
title('RNN for Classification')
test_y=grp2idx(test_y);
YPred1=grp2idx(YPred1);
cp2 = classperf(test_y,YPred1)
[confusionMat] = confusionmat(test_y,YPred1);
[confMat,order]=confusionmat(test_y,YPred1);

acc21 = cp2.CorrectRate;
disp(['Accuracy = ',num2str(acc21*100),'%']);

disp(['Error Rate = ',num2str((1-acc21)*100),'%']);

Sensitivity2 = cp2.Sensitivity;
disp(['Sensitivity = ',num2str(Sensitivity2*100),'%']);

Specificity2 = cp2.Specificity;
disp(['Specificity = ',num2str(Specificity2*100),'%']);



