
% @author: antriksh
% Version 0: 2/25/2018

function [mitoc, mctoi] = messages(n, k, A, w, its, type)
    mitoc = ones(n, n, k); % Messages Clique to i
    mctoi = ones(n, n, k); % Messages i to Clique
    
%     LOOP FOR NUMBER OF INTERATIONS
    for it = 1:its
        disp("Iteration: " + it);
        
%         UPDATING MESSAGES FROM CLIQUE TO VARIABLES
        msg = zeros(size(mctoi));
        for i = 1:n
            edges = getEdges(A, i);
            for j = edges
                
                % MESSAGE FROM ALL OTHER EDGES TO CLIQUE
                mOthersToC = ones(1, k);
                sizeClique = size(edges, 2);
                for clis = edges(edges ~= j)
                    mOthersToC = mOthersToC .* squeeze(mitoc(i, clis, :))';
                end
                
                if sizeClique > k
                    disp("Size of clique is greater than number of colors, Unacceptable !");
                    return;
                end
                
                cartProd = cartProduct(sizeClique, k);
                
                for index = 1:length(cartProd)
                    colors = cartProd(:, index);
                    amax = all(diff(sort(colors(:))));
                    if type == 'sum'
                        msg(i, j, colors(end)) = msg(i, j, colors(end)) + sum(amax .* mOthersToC);
                    elseif type == 'max'
                        msg(i, j, colors(end)) = msg(i, j, colors(end)) + max(amax .* mOthersToC);
                    end
                end
                
                mctoi(i, j, :) = normalize(msg(i, j, :));
%                 disp("mctoi " + i + "," + j + ": " + squeeze(mctoi(i, j, :)));
            end
        end
        
%         UPDATING MESSAGES FROM VARIABLES
        for i = 1:n
            clique = getEdges(A, i);
            for j = clique
                otherCliques = getEdges(A, j);
                otherCliques = otherCliques(otherCliques ~= i);
                msgOtherCliques = ones(1, k);
                sizeClique = size(otherCliques, 2);
                
                for c = otherCliques
                    msgOtherCliques = msgOtherCliques .* squeeze(mctoi(i, c, :))';
                end
                
                mitoc(i, j, :) = normalize(exp(w) .* squeeze(msgOtherCliques));
%                 disp("mitoc " + i + "," + j + ": " + squeeze(mitoc(i, j, :)));
            end
        end
    end
    
end