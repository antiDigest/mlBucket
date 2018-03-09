
% @author: antriksh
% Version 0: 2/28/2018
% Version 1: 3/5/2018
%     Added random selection function
%     Added initialization to a single clique with different colors
    

function m = gibbs(A, w, burnin, its)

    function print()
        d = [];
        for edge = E
            i = edge(1);
            j = edge(2);
            d = [d; edgeValues(i, j)];
        end
        d'
    end

    a = 1;
    d = 4;
    adcount = 0;
    allCount = 0;

%     disp(A); % DISPLAY MATRIX
    
    [~, k] = size(w); % Number of colors {1, .. , k}
    [N, ~] = size(A); % Size of (rows and columns) Adjacency Matrix of graph
%     its = burnin + its; % Total iterations

    counts = zeros(N, N, k);
    m = zeros(N, N, k); % Initialize m
    E = getEdges(A);
    edgeValues = zeros(N, N);
    
    colors = [2 4 1 3 4]; % The edge color values have been hard coded to [2 4 1 3 4] 
    for index = 1:length(colors)
        e = E(:, index);
        i = e(1);
        j = e(2);
        edgeValues(i, j) = colors(index);
    end
    
    edgeValues
    
    for it = 1:burnin
%         disp("Burnin Iteration: " + it);
        
        for e = E
            i = e(1);
            j = e(2);
            probs = [];
            
            % Probability for e being any of 1:k
            for color = 1:k
                [phi, psi] = probability(A, N, E, e, color, w, edgeValues);
                probs = [probs phi * psi];
            end
            probs = normalize(probs);
            edgeValues(i, j) = selectRandom(probs);
        end
    end
    
    for it = 1:its
%         disp("Iteration: " + it);
        
        for e = E
            i = e(1);
            j = e(2);
            probs = [];
            
            % Probability for e being any of 1:k
            for color = 1:k
                [phi, psi] = probability(A, N, E, e, color, w, edgeValues);
                probs = [probs phi * psi];
            end
            probs = normalize(probs);
            m(i, j, :) = probs;
            edgeValues(i, j) = selectRandom(probs);
            counts(i, j, edgeValues(i, j)) = counts(i, j, edgeValues(i, j)) + 1;
        end
    end
    
    for e = E
        i = e(1);
        j = e(2);
        counts(i, j, :) = normalize(counts(i, j, :));
    end
    
    probad4 = counts(a, d, 4);
    disp("burnin: " + burnin + "; its: " + its);
    disp("Edge (a,d), color=4: " + probad4);
    
end