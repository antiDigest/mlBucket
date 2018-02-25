
% @author: antriksh
% Version 0: 2/25/2018

function [j, index] = join(A, a, B, b)
    q = unique([A(:,a), B(:,b)]', 'rows')';
    aInd = ismember(q, A(:, a));
    bInd = ismember(q, B(:, b));
    x = zeros(length(q), size(A, 2) + size(B, 2) - 1);
    x(:, a) = q;
    x(aInd, 1: a - 1) = A(:, 1: a - 1);
    x(aInd, a + 1: size(A, 2)) = A(:, a + 1: size(A, 2));
    x(bInd, size(A, 2) + 1: size(A, 2) + b - 1) = B(:, 1: b - 1);
    x(bInd, size(A, 2) + b: end) = B(:, b + 1: end);
    j = x;
    index = a;
end