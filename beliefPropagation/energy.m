
% @author: antriksh
% Version 0: 2/25/2018


function e = energy(A, n, k, bc, bi)

    %     BETHE FREE ENERGY EQUATION
    z2 = zeros(1, n);
    for i = 1:n
        clique = getEdges(A, i);
        sizeClique = size(clique, 2);
        if sizeClique > 1
            
            cartProd = cartProduct(sizeClique, k);
            ks = k * ones(1, sizeClique);
            amax = zeros(ks);
            for index = 1:length(cartProd)
                colors = cartProd(:, index);
%                 values = mat2cell(colors', 1, sizeClique);
                amax(index) = all(diff(sort(colors)));
            end
            if amax
                x = log(amax .^ bc{i});
            else
                x = 0;
            end
            z2(i) = z2(i) + sum(x(:));
        end
    end
    
    e = entropy(A, n, bi, bc, k) + sum(z2(:));
end