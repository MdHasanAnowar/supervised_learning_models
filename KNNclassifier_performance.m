clc;
clear all;

%the code generates random data each time it runs
%so, each time generates a different output

%reading the data from the file and making a data table 
all_data_table=readtable('iris.dat'); 

k=1:8; %choosing k from 1 to 8
iters=100; %number of iteration demanded
accuracies_per_k_per_iters=zeros(length(k),iters); %defining accuracy per k per iteration with all zeros
average_accuracy_per_k=zeros(length(k),1); %initializing the average accuracy per k matrix with all zeros
standard_deviation_per_k=zeros(length(k),1); %initializing the standard deviation per k matrix with all zeros 

%have to split all the data into two class- test class and training class
%test class contains 30% of all data- that is 45 test data
%training class contains 70% of all data- that is 105 data

for i=1:iters

    %creating training data set and test data set 
    
    %the following code generates random data each time it run
    %so, each time generates a different output
  
    X = table2array(all_data_table(:,1:4)); %separating all data
    y = all_data_table(:,5).Var5; %separating all labels
    idx = (randperm(length(y)))'; %creating a matrix of randomized number from 1 to 150 
    training_data = X(idx(1:105),:); %creating training data
    training_labels = y(idx(1:105)); %creating training labels
    test_data = X(idx(106:150),:); %creating test data
    test_labels = y(idx(106:150)); %registering actual labels of training data

    %Measuring performance

    %calculating accuracy for each selection of kNN
    accuracy=zeros(1,length(k)); %initializing the accuracy matrix 
    for tally=1:length(k) %following code snippet iters for k times
        [pred_labels]=knnclassify(test_data,training_data,training_labels,tally); %calling the knnclassify function
        accuracy(tally)=100*(sum(pred_labels==test_labels)/length(test_labels)); %calculating accuracy  
    end

accuracies_per_k_per_iters(:,i)=accuracy; %allocating accuracy per k per iteration  

end

%calculating average accuracy and standard deviation per k
for tally=1:length(k) %following code snippet iters for k times
    average_accuracy_per_k(tally,1)=(sum(accuracies_per_k_per_iters(tally,:)))/iters; %calculating average accuracy per k
    standard_deviation_per_k(tally,1)=sqrt(mean((average_accuracy_per_k(tally,1)-accuracies_per_k_per_iters(tally,:)).^2)); %calculating standard deviation per k
    fprintf('Average classification accuracy for for k=%d is %f percent \n', tally,average_accuracy_per_k(tally,1));
    fprintf('Standard deviation for for k=%d is %f \n', tally,standard_deviation_per_k(tally,1));
    fprintf('\n');
end

%plotting the average accuracy curve and  the bar chart of standard deviation per k
errorbar(k,average_accuracy_per_k,standard_deviation_per_k,'-s','MarkerSize',5,'MarkerFaceColor','k');
grid on;
xlabel('Number of nearest neighbour k in consideration');
ylabel('Average accuracy in percentage with standard deviation');
axis ([0 length(k)+1 88 100]); %choosing axis limit
title('Performance of k-NN classifier per k with error bar standard deviation');
print('k-NN classifier performance','-djpeg','-r800');

%end of the code


