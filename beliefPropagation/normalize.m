
% @author: antriksh
% Version 0: 2/20/2018

function f = normalize(A)
    if sum(A(:)) ~= 0
        f = (A)./(sum(A(:)));
    else
        f = zeros(size(A));
    end
end
