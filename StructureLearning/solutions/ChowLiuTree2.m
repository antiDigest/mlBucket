%Learn a discriminative Bayesian network for the class label (the first column) 
%by using the Chow-Liu Bayesian structure learning algorithm 

function cl = ChowLiuTree2()    
    trainfile = 'mushroom_train.data';
    testfile = 'mushroom_test.data';
    format = '%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c';
    train_data = cell2mat(textscan(fopen(trainfile,'rt'),format));
    test_data = cell2mat(textscan(fopen(testfile,'rt'),format));
    %Create N to store the size of the training data
    %Create dim to store the number of attributes
    [N, dim] = size(train_data);
    %Calculating the uni-marginal distribution
    unimarginals = cellfun(@(x) marginal(N, x), cellstr(train_data'));
    %Store the marginal distribution of label
    labelMarginal = unimarginals(1);
    mutualInfo = zeros(dim,dim);
    for s=1:dim
        for t=s+1:dim
            pairm = pairMaginal(N,train_data(:,s),train_data(:,t));
            minfo = 0;
            for j=1:numel(pairm.x1)
                for k=1:numel(pairm.x2)
                    pmg = pairm.dist(j,k);
                    if pmg~=0
                        minfo = minfo + pmg*log(pmg/(unimarginals(s).dist(unimarginals(s).x==pairm.x1(j)) * unimarginals(t).dist(unimarginals(t).x==pairm.x2(k))));
                    end
                end
            end
            mutualInfo(s,t) = minfo;
            mutualInfo(t,s) = minfo;
        end
    end
    %build Chow LiuTree
    tree = minspantree(graph(-mutualInfo));
    plot(tree);
    %the neighbors of the root node
    neibrs = neighbors(tree,1);
    %store the pair marginals of label node and its neighbours
    labelPairMarginals = arrayfun(@(n)pairMaginal(n, train_data(:,1),train_data(:,n)), neibrs);
    %the degree of root node
    degree = numel(neibrs);
    %predict the result
    predictions = cellfun(@(x) predict(labelMarginal, labelPairMarginals, neibrs, N, degree,x),cellstr(test_data));
    accuracy = sum(test_data(:,1)==predictions)/size(test_data,1);
    fprintf('The test accuracy of Chow-Liu tree classifier is: %f\n',accuracy); 
end
%Calculating the uni-marginal distribution
function unimarginals = marginal(N,X)
    unimarginals.x = unique(X);
    unimarginals.dist = arrayfun(@(x)sum(X==x)/N,unimarginals.x);
end
% Calculating the pair-marginal distribution
function pairm = pairMaginal(N,X1,X2) 
    pairm.x1 = unique(X1);
    pairm.x2 = unique(X2);
    pairm.dist = zeros(numel(pairm.x1),numel(pairm.x2));
    for i=1:numel(pairm.x1)
        for j=1:length(pairm.x2)
            pairm.dist(i,j) = sum(X1==pairm.x1(i) & X2==pairm.x2(j))/N;
        end
    end
end
% predict y-label using the model; 
function y = predict(labelMarginal, labelPairMarginals, neibrs, N, degree, X)
    score = zeros(1,numel(labelMarginal.x));
    X_s = X(neibrs);
    for i = 1:length(score)
        score(i) = (1 - degree)*log(labelMarginal.dist(i));
        for j=1:degree
            pm = labelPairMarginals(j);
            pb = pm.dist(pm.x1==labelMarginal.x(i),pm.x2==X_s(j));
            if pb ==0 
                pb = numel(pm.dist)/(numel(pm.dist)+N);
            end
            score(i) = score(i) + log(pb);
        end
    end
    [~,idx] = max(score);
    y = labelMarginal.x(idx);
end
        

