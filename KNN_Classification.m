data = readtable('D:\Training\Classification\K-Nearest Neighbor\Social_Network_Ads.csv'); %load dataset

complete_data = rmmissing(data) %data preprocessing, remove missing value

%Feature scaling%

standardize_age = (data.Age - mean(data.Age))/std(data.Age)
data.Age = standardize_age;

standardize_estimatedsalary = (data.EstimatedSalary - mean(data.EstimatedSalary))/std(data.EstimatedSalary)
data.EstimatedSalary = standardize_estimatedsalary;

%Classifying Data%
classification_modelKNN = fitcknn(data, 'Purchased~Age+EstimatedSalary');


%test and train%
cv = cvpartition(classification_modelKNN.NumObservations, 'HoldOut', 0.2);
cross_validated_model = crossval(classification_modelKNN, 'cvpartition', cv)

%making prediction%
Predictions = predict(cross_validated_model.Trained{1}, data(test(cv), 1:end-1));

%analyzing the predicition%
Results = confusionmat(cross_validated_model.Y(test(cv)), Predictions);

%visualizing training set results%

labels = unique(data.Purchased);
classifier_name = 'K-Nearest Neighbor';

Age_range = min (data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];

predictions_meshgrid = predict(cross_validated_model.Trained{1}, XGrid);

gscatter(xx1(:), xx2(:), predictions_meshgrid, 'rgb');

hold on

training_data = data (training(cv),:);
Y = ismember(training_data.Purchased, labels{1});

scatter(training_data.Age(Y), training_data.EstimatedSalary(Y), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Age(~Y), training_data.EstimatedSalary(~Y), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');

xlabel('Age');
ylabel('Estimated Salary');

title(classifier_name);