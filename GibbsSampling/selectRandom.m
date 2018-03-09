
% @author: antriksh
% Version 0: 3/5/2018

function r = selectRandom(probs)
    cumProbs = cumsum(probs);
    random = rand;
    r = 1;
    for index = 1:length(cumProbs)
        if index < length(cumProbs) && ...
            cumProbs(index) < random && ...
            cumProbs(index + 1) > random
            r = index + 1;
            break;
        elseif index == length(cumProbs) && ...
                cumProbs(index) < random
            r = index;
            break;
        end
    end
end
