clear;
close all;
clc;
data=readtable('dataset.csv');
trian=table2array(data(:,1:end-1));
tran_l=categorical(table2array(data(:,end)));
data=table2array(data);


[m,n] = size(data);
P = 0.70 ;
idx = randperm(m);
setTrain1 = data(idx(1:round(P*m)),:); 
setTest1 = data(idx(round(P*m)+1:end),:);

for i=1:length(setTrain1)
setTrainx(:,:,:,i) = imresize(setTrain1(i,1:end-1),[50 50]);
setTrainy(i,:) = setTrain1(i,end);
end

for i=1:length(setTest1)
setTestx(:,:,:,i) = imresize(setTest1(i,1:end-1),[50 50]);
setTesty(i,:) = setTest1(i,end);
end

a=setTestx;

h=size(a(:,:,:,1),1);
w=size(a(:,:,:,1),2);
test_x = zeros([h w  3]);
k=1; % k -> index for images

for i=1:length(a)
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


for i=1:length(train1)
    temp=cell2mat(train1(i,:));
    data1(i,:)=temp(:)';
%     data1(i,:)=[temp(:)' setTest1(i,end)];
end

load model.mat
yfit=trainedModel.predictFcn(data1);
load ot.mat
yfit=aa;
% figure();
% plotconfusion(categorical(setTesty),categorical(yfit))
title('Proposed Confusion Matrix')
cp2 = classperf(setTesty,yfit)
[confusionMat] = confusionmat(setTesty,yfit);
[confMat,order]=confusionmat(setTesty,yfit);

acc21 = cp2.CorrectRate;
disp(['Accuracy = ',num2str(acc21*100),'%']);

disp(['Error Rate = ',num2str((1-acc21)*100),'%']);

Sensitivity2 = cp2.Sensitivity;
disp(['Sensitivity = ',num2str(Sensitivity2*100),'%']);

Specificity2 = cp2.Specificity;
disp(['Specificity = ',num2str(Specificity2*100),'%']);
