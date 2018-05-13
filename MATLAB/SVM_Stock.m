features = csvread("nonPercentFeatures.csv");


last = features(1,1);
for i = 1:length(features)
    tempLast = features(i,1);
    features(i,1) = features(i,1)/last;
    last = tempLast;
end

features(5)

labels = features(2:end,1);

features = features(1:end-1,:);





featureSize = size(features);
labelSize = size(labels);


assert(featureSize(1) == labelSize(1))

train_features = features(1:0.8*featureSize(1),:);
train_y = labels(1:0.8*labelSize(1),:);

svm = fitrsvm(features,labels,'Standardize',true,'KernelFunction','rbf')

svm.ConvergenceInfo.Converged
iter = svm.NumIterations

test_features = features(0.8*featureSize(1):end,:);
test_y = labels(0.8*labelSize(1):end,:);



pred = predict(svm, test_features);
time = (1:length(test_y));
length(time)
length(test_y)
length(pred)
legend()

plot(time, test_y)
hold on;
plot(time, pred)
hold on;
legend('Real Value','Predicted');
total = length(test_y);
rightCount = 0;

for i = 1:length(test_y)
    if(sign(pred(i) - 1) == sign(test_y(i) - 1))
        rightCount = rightCount + 1;
    end
end
rightCount/total