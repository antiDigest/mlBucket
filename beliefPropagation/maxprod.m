
% @author: antriksh
% Version 0: 2/18/2018

function X = maxprod(A, w, its)
    
    k = size(w, 2); % Number of colors {1, .. , k}
    [m,n]= size(A); % Size of (rows and columns) Adjacency Matrix of graph
    Vertices = 1:m; % Vertices of graph
    
    [mitoc, mctoi] = messages(n, k, A, w, its, 'max');
    
    [bi, bc] = beliefs(n, k, A, w, mitoc, mctoi);
    
%     CALCULATING X FOR EACH EDGE
    X = zeros(n, n);
    for i = 1:n
        clique = getEdges(A, i);
        sizeClique = size(clique, 2);
        if sizeClique > 1
            belief = bc{i};
            mx = max(belief(:));
            argmax = find(belief == mx)';

            if size(argmax, 2) > 1 | isempty(argmax)
                for j = clique
                    X(i, j) = 0;
                    X(j, i) = 0;
                end
                continue;
            end

            arg = cell(1, sizeClique);
            [arg{:}] = ind2sub(size(belief), argmax)

            ind = 1;
            for j = clique
                if X(i, j) ~= 0 && X(i, j) ~= arg{ind}
                    X(i, j) = 0;
                    X(j, i) = 0;
                else
                    arg{ind}
                    X(i, j) = arg{ind};
                    X(j, i) = arg{ind};
                    ind = ind + 1;
                end
            end
        else
            for j = clique
                belief = bi(i, j, :);
                mx = max(belief(:));
                argmax = find(belief == mx)';

                X(i, j) = argmax;
                X(j, i) = argmax;
            end
        end
    end
end