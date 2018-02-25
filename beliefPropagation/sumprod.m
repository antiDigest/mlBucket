
% @author: antriksh
% Version 0: 2/18/2018
% Version 1: 2/25/2018
%     Made separate functions: messages, beliefs and energy

function Z = sumprod(A, w, its)
    function [x, y] = swap(x1, y1)
        x = x1;
        y = y1;
    end

%     disp(A); % DISPLAY MATRIX
    
    k = size(w, 2); % Number of colors {1, .. , k}
    [m,n]= size(A); % Size of (rows and columns) Adjacency Matrix of graph
    Vertices = 1:m; % Vertices of graph

    [mitoc, mctoi] = messages(n, k, A, w, its);
    
    [bi, bc] = beliefs(n, k, A, w, mitoc, mctoi);
    
    Z = energy(A, n, k, bc, bi);
end