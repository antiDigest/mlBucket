
% @author: antriksh
% Version 0: 3/2/2018

function [phi, psi] = probability(A, N, E, node, nodeVal, w, edgeValues)

    pv = [w(nodeVal)];
    psi = 1;
    
    for e = E
        if e(1) == node(1) && e(2) == node(2)
            continue;
        end
        i = e(1);
        j = e(2);
        pv = [pv exp(w(edgeValues(i, j)))];
    end
    phi = prod(pv);

    psiValues = [];
    for i = 1:N
        clique = edgesToi(A, i);

        % for each edge in the clique, find out psi
        pv = [];
        for c = clique
            if (i == node(1) && c == node(2)) || ...
                    (i == node(2) && c == node(1))
                pv = [pv nodeVal];
            elseif ismember([i c], E', 'rows')
                pv = [pv edgeValues(i, c)];
            else
                pv = [pv edgeValues(c, i)];
            end
        end
        psiValues = [psiValues all(diff(sort(pv(:))))];
    end
    psi = prod(psiValues);
end