%%%Implementation the K-NN Classifier%%%

%creating a function that will predict class of famous iris dataset
%name of the function is knnclassify
%test_data,training_data,training_labels & k are the inputs
%pred_labels is the output

%calling the fucntion with inputs and outputs
function[pred_labels]=knnclassify(test_data,training_data,training_labels,k)
    pred_labels=strings(length(test_data),1); %defining the predicted labels with blank string
    l_error='2NN error';
    %creating the set of labels that is to be tagged on training data
    char_available_labels={'Iris-setosa','Iris-versicolor','Iris-virginica'}; %creating a char matix of available labels
    available_labels=string(char_available_labels); %creating a string matix of available labels
    
    for i=1:1:length(test_data) %the following code snippet iters for each of the test data
    
    %calculating the eucledian distance between the training and test data
        dis =((test_data(i,:)-training_data)).^2;   
        dist = sqrt(sum(dis,2));
    
    %sorting out the indices of k lowest diastances between data points
        dist_sort=sort(dist); %sorting out the dist matrix
        mins=(dist_sort(1:k,1)); %choosing the k lowest distances
        index=find((dist<=max(mins)))'; %finding the indices of the minimum distances from the dist matix
    
    %counting nearest neighbours and predicting the class
        count=zeros(1,length(available_labels)); %creating a zero matrix for counting purpose
            for j=1:k %the following code snippet iters for k times
                count=count+strcmp(training_labels(index(j)),available_labels);
                %counting how many times each of the available test labels match with the training labels
            end
        l_index=find(count==max(count)); %choosing the index of the maximum match from the available test labels
        if length(l_index)==1
            pred_labels(i,1)=available_labels(l_index); %labeling the test points similar with the maximum match
        else
            pred_labels(i,1)=string(l_error); %2 nearest neighbour located, so counted as error 
        end
    end
end

%end of the function