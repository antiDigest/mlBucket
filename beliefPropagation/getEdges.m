
% @author: antriksh
% Version 0: 2/20/2018

function e = getEdges(A, i)
    v = A(1:end, i) == 1;
    e = find(v == 1);
    e = e';
end
