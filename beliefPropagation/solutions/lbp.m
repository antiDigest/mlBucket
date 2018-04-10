% (loopy) belief propagation for generic factor graph;
% custom psi (singleton potential) and phi (clique potential) can be
% defined for general problems; here we focus on the edge coloring problem
% Yibo Yang, March 2018

function [beliefs, factor_graph] = lbp(A, w, its)
op = @sum;
% define potential functions for the coloring problem
phi = exp(w);
% clique potential for coloring problem; 1 if no color clash, 0 o/w
% psi = @(cfg) length(unique(cfg)) == length(cfg);

factor_graph = convert_to_factor_graph(A);
N = factor_graph.N;
K = length(w);
neighbors=factor_graph.neighbors;
M = ones(N, N, K);  % central message storage; details see calc_message
for t=1:its
    for i=1:N
        for j=neighbors{i}
            M(i,j,:) = calc_message(factor_graph, phi, @psi, M, i, j, op);
        end
    end
end

% calc beliefs (approximate marginals)

% singleton beliefs; equivalent to the corresponding singleton clique marginals in clique_marg
E = factor_graph.E;
node_marg = ones(E, K);
for e=1:E   % node id counts from 1
    nbr_msg = ones(K, 1);
    for j=neighbors{e}  % mult all incoming messages (including from singleton cliques)
        nbr_msg = nbr_msg .* squeeze(M(j, e, :));
    end
    nbr_msg = nbr_msg/sum(nbr_msg);
    node_marg(e,:) = nbr_msg';
end

num_nonsingleton_cliques = length(factor_graph.orig_connecting_verticies);
D = E + num_nonsingleton_cliques;  % D==N-E
% beliefs is a Dx1 cell array; the first E entries store node marginals
% (same as singleton clique marginals) in the order of
% factor_graph.orig_edges and the rest num_nonsingleton_cliques entries
% store (non-singleton) clique marginals in the order of 
% factor_graph.orig_connecting_verticies;
% the node marginal correspond to [P(x=1); P(x=2); ... P(x=k)];
% the clique marginal correspond to (say clique_size=2, K=3):
% [P(x1=1,x2=1); P(x1=1,x2=2); P(x1=1,x2=3); P(x1=2,x2=1); P(x1=2,x2=2);...
% P(x1=3,x2=3)], i.e., each clique marginal is a K^clique_size vector whose
% elements correspond to joint clique configurations in sorted order

beliefs = cell(D, 1);
for e=1:E
    beliefs{e} = node_marg(e,:);  % copying over
end

% clique beliefs
% here we only compute nonsingleton clique marginals, as we've already
% computed singleton clique (node) marignals above [which can also be
% computed here, if we want, as `c_marg = phi(:) .* squeeze(M(nbrs, c, :));
% c_marg = c_marg / sum(c_marg)` where c is the id of a singleton clique
for c=(E+1):D   % nonsingleton clique id
    nbrs = neighbors{c};
    clique_size=length(nbrs);
    assert(clique_size>1, 'should be a nonsingleton clique!');
    clique_config = permn(1:K, clique_size); % 111, 112, 113, 121, 122, 123, ...
    num_config = length(clique_config);
    c_marg = zeros(1, num_config);
    for n=1:num_config
        cfg = clique_config(n, :);
        prod = psi(cfg);    % multiply by clique potential
        if prod==1
            for nbr_idx=1:clique_size
                nbr_id = nbrs(nbr_idx);
                nbr_cfg = cfg(nbr_idx);
                prod = prod * M(nbr_id, c, nbr_cfg);
            end
        end
        c_marg(n) = prod;
    end
    c_marg = c_marg / sum(c_marg);
    beliefs{c} = c_marg;
end

end



function good = psi(cfg)
    % clique potential for coloring problem; 1 if no color clash, 0 o/w
    good = length(unique(cfg)) == length(cfg);
end