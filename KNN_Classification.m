data = readtable('D:\Working Documents\tMatlab-ML\Social_Network_Ads.csv'); %load dataset

complete_data = rmmissing(data) %data preprocessing, remove missing value

%Feature scaling%

standardize_age = (data.Age - mean(data.Age))/std(data.Age)
data.Age = standardize_age;

standardize_estimatedsalary = (data.EstimatedSalary - mean(data.EstimatedSalary))/std(data.EstimatedSalary)
data.EstimatedSalary = standardize_estimatedsalary;

%Classifying Data%

classification_modelKNN = fitcknn(data, 'Purchased~Age+EstimatedSalary');