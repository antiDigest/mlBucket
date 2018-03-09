
% @author: antriksh
% Version 0: 2/20/2018

function e = getEdges(A)
    edges = find( triu(A > 0) );
    e = [];
    for index = 1:length(edges)
        [i, j] = ind2sub(size(A), edges(index)); % node indices of edge e  
        e = [e; i j];
    end
    e = e';
end
