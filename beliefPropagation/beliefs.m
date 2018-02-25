
%  @author: antriksh
% Version 0: 2/25/2018

function [bi, bc] = beliefs(n, k, A, w, mitoc, mctoi)
         
%     VARIABLE BELIEFS
    bi = zeros(n, n, k);
    for i = 1:n
        clique = getEdges(A, i);
        for j = clique
            bi(i, j, :) = normalize(exp(w) .* squeeze(prod(mctoi(i, clique(:), :)))');
        end
    end

%     CLIQUE BELIEFS
    bc = cell(1, n);
    for i = 1:n
        clique = getEdges(A, i);
        
        sizeClique = size(clique, 2);
        if sizeClique > 1
            ks = k * ones(1, sizeClique);

            msg = zeros(ks);
            if size(ks, 2) == 1
                msg = ones(1, ks);
            end

            bc{i} = msg;

            m = ones(1, k);
            x = k; y = 1;
            for edge = clique
                m = m .* squeeze(mitoc(i, edge, :))';
            end
            
            cartProd = cartProduct(sizeClique, k);
            ks = k * ones(1, sizeClique);
            amax = zeros(ks);
            for index = 1:length(cartProd)
                colors = cartProd(:, index);
%                 values = mat2cell(colors', 1, sizeClique);
                amax(index) = all(diff(sort(colors)));
            end
            
            bc{i} = bc{i} + amax .* m;

            bc{i} = normalize(bc{i});
%             disp("Bc " + i + ",: " + bc{i});
        end
    end
    
end