
% @author: Antriksh
% Version 0: 4/17/2018

function naive_bayes()
    A = readtable('mushroom_train.csv', 'ReadVariableNames', false);
    
    [R, C] = size(A);
    
    classes = unique(A.Var1);
    priorProb = zeros(1, size(classes, 1));
    
    for i = 1:size(classes, 1)
        priorProb(i) = sum(strcmp(A.(1), classes{i}));
    end
    priorProb = priorProb / R;
    
    Mdl = fitcnb(A(:, 2:end), A(:, 1));
    Mdl.Prior, priorProb
    
    TEST = readtable('mushroom_test.csv', 'ReadVariableNames', false);
%     TEST
% %     TEST.(5){2}
    [RT, CT] = size(TEST);
    post = zeros(RT, CT);
    acc = 0.0;
    total = 0.0;
    store = [];
    for ro = 1:RT
        post = ones(1, size(classes, 1));
        for class = 1:size(classes, 1)
            p = post(class) * priorProb(class);
            for c = 2:CT % First column is class variable
                p = p * probability(c, TEST.(c){ro}, 1, classes{class});
            end
            post(class) = p;
        end
        post = post / sum(post);
        posterior(ro, :) = post;
        [argvalue, argmax] = max(post);
        store = [store {classes{argmax}}];
%         TEST.(1){ro}, classes{argmax}
        if strcmp(TEST.(1){ro}, classes{argmax})
            acc = acc + 1.0;
        end
        total = total + 1.0;
    end

    accuracy = acc / total;
    
    [predictedspecies, Posterior, ~] = predict(Mdl, TEST(:,2:end));
%     Posterior(89:100, :)
%     posterior(89:100, :)
    accuracy
    
    function p = probability(column1, value1, column2, value2)
        a = strcmp(A.(column1), value1);
        b = strcmp(A.(column2), value2);
        single = sum(strcmp(A.(column2), value2));
        
        combined = 0;
        for r = 1:R
            if a(r)
                combined = combined + 1.0 * (a(r) == b(r));
            end
        end
        
        if combined ~= 0
            p = combined / single;
        else
            p = 1 / single;
        end
    end 
    
end