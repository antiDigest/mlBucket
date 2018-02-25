
% @author: antriksh
% Version 0: 2/25/2018

function args = maxArgs(bcmat)
    value = max(bcmat);
    argmaxs = find(bcmat == value & bcmat > 0)';
    v = max(bcmat(argmaxs));
    args = find(bcmat == v)';
end