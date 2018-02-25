
% @author: antriksh
% Version 0: 2/23/2018
% Version 1: 2/24/2018
%     Producing cartesian product of k colors for sizeClique elements
%     Changed from Producing cartesian product of k colors for k elements

function f = cartProduct(sizeClique, k)
    s = repmat({1:k}, 1, sizeClique);
    [Q{1:sizeClique}] = ndgrid(s{:});
    f = reshape(cat(k ,Q{:}), [], sizeClique)';
end
