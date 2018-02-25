
% @author: antriksh
% Version 0: 2/23/2018

%     CALCULATING ENTROPY USING Taoi = bi and TaoC = bc
function h = entropy(A, n, bi, bc, k)

    hbi = zeros(n, n);
    hbc = zeros(1, n);

    for i = 1:n
        clique = getEdges(A, i);
        for j = clique
            hbi(i, j) = hbi(i, j) + sum(log(squeeze(bi(i, j, :)) .^ squeeze(bi(i, j, :))));
        end
    end

    for i = 1:n

        hbc(i) = 0;

        clique = getEdges(A, i);
        sizeClique = size(clique, 2);
        if sizeClique > 1
            
            x = log((bc{i}./squeeze(prod(bi(i, clique(:), :), 2))') .^ bc{i});

            hbc(i) = hbc(i) + sum(x(:)); 
        end
    end
    
    h = - sum(hbc(:)) - sum(hbi(:));
end