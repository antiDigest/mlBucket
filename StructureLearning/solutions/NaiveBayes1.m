% Train a naive Bayes classifier to predict.
function nb = NaiveBayes1()
    clear;
    trainfile = 'mushroom_train.data';
    testfile = 'mushroom_test.data';
    format = '%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c';
    train_data = cell2mat(textscan(fopen(trainfile,'rt'),format));
    test_data = cell2mat(textscan(fopen(testfile,'rt'),format));
    %Constructor of model; 
    %Using the matrix to create the model, column is the feature of the model, 
    %row is the training data, y_postion is the column index of label
    %Create the train_y to store y training data
    train_y = train_data(:,1);
    %Create the train_x to store X training data
    train_x = train_data(:,(2:end));
    %Create label
    label = unique(train_y);
    % Training a naive bayes classifier 
    % Calculation each conditional distribution
    distribute = arrayfun(@(y)sum(train_y==y),label);
    marginal = distribute/sum(distribute);
    rows = numel(label);
    cols = size(train_x,2);
    model = cell(rows,cols);
    for i=1:cols
        col = train_x(:,i);
        for j=1:rows
            model{j,i}.feature = unique(col);
            model{j,i}.prob = arrayfun(@(x) sum(col(train_y==label(j))==x)/distribute(j), model{j,i}.feature);
        end
    end
    % predict y-label using the model;
    predictions = cellfun(@(x) predict(model, label, marginal, x),cellstr(test_data(:,2:end)));
    accuracy = sum(test_data(:,1)==predictions)/size(test_data,1);
    fprintf('The test accuracy of Naive Bayes classifier is: %f\n',accuracy);            
end

% predict y-label using the model; 
function y = predict(model, label, marginal,X)
    score = zeros(1,numel(label));
    for i = 1:length(score)
        score(i) = log(marginal(i)) + sum(log(arrayfun(@(m,x) m.prob(m.feature==x),[model{i,:}],X)));
    end
    [~,idx] = max(score);
    y = label(idx);
end
   


