
% Learn the structure of the Bayesian Network using Chow-Liu Trees
% @author: Antriksh
% Version 0: 04/21/2018

function [dmst, parents] = chow_liu_trees()
    
    A = readtable('mushroom_train.csv');
    [R, C] = size(A);
    W = mutual_info();
    parents = zeros(1, C);
    
    [dmst, cost] = UndirectedMaximumSpanningTree(W);

    for i = 1:C
        node.value = i;
        node.visited = false;
        nodes(i) = node;
    end
    
    root = nodes(1);
    parent = root;
    nodeCount = 0;
    while nodeCount < C
        nodeCount = nodeCount + 1;
        parent = nodes(nodeCount);
        [parents, nodeCount] = dfs(parents, nodes, nodeCount, parent);
        parents(parent.value) = 0;
    end
    
    function [parents, nodeCount] = dfs(parents, nodes, nodeCount, parent)
        
        parent.visited = true;
        
        children = find(dmst(parent.value, :));
        for j = children
            child = nodes(j);
            if ~child.visited && ~parents(child.value)
                if parents(child.value) ~= parent.value
                    parents(child.value) = parent.value;
                    [parents, nodeCount] = dfs(parents, nodes, nodeCount + 1, child);
                end
            end
        end
    end
    
    for i = 1:C
        for j = 1:C
            if parents(j) == i
                dmst(i, j) = W(i, j) * dmst(i, j);
            else
                dmst(i, j) = 0;
            end
        end
    end
    
    h1 = view(biograph( dmst, [1:23], 'ShowArrows', 'on', 'ShowWeights', 'on' ));    
    
    
    function I = mutual_info()
        I = zeros(C, C);
        for i = 1:C
            unique_i = cell2mat(unique(A.(i)));
            for j = 1:i-1
                unique_j = cell2mat(unique(A.(j)));
                [p,q] = meshgrid(unique_i, unique_j);
                pairs = [p(:) q(:)];
                unique_ij = unique(sort(pairs, 2), 'rows');
                for ij = unique_ij
                    val_i = ij(1);
                    val_j = ij(2);
                    pxixj = joint_prob(i, val_i, j, val_j);
                    pxi = prob(i, val_i);
                    pxj = prob(j, val_j);
                    if pxixj ~= 0
                        I(i, j) = I(i, j) + pxixj * log(pxixj / (pxi * pxj));
                        I(j, i) = I(j, i) + pxixj * log(pxixj / (pxi * pxj));
                    end
                end
                %                 I(i,j) = I(i, j);
            end
        end
    end
    
    function Pxixj = joint_prob(column1, value1, column2, value2)
        a = strcmp(A.(column1), value1);
        b = strcmp(A.(column2), value2);
        
        combined = 0;
        for r = 1:R
            if a(r)
                combined = combined + 1.0 * (a(r) == b(r));
            end
        end
        
        Pxixj = combined / R;
    end
    
    function Pxi = prob(column, value)
        Pxi = sum(strcmp(A.(column), value)) / R;
    end
    
end