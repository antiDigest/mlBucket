
% @author: antriksh
% Version 0: 2/25/2018

function colors = getColors(sizeClique, argmax, bcmat)
    colors = zeros(size(argmax, 1), sizeClique);
    for index = 1:length(argmax)
        arg = cell(1, sizeClique);
        [arg{:}] = ind2sub(size(bcmat), argmax(index));
        colors(index, :) = cell2mat(arg);
    end
end