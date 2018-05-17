% Same as lbp2, but we allow conditioning on a subset of random variables
% to perform conditional inference p(x_C|x_obs). The only change we make is
% (conceptually) multiplying the potential function phi_i(x_i) of every
% observed r.v., X_i=x_i', with the indicator function 1_{x_i==x_i'} that
% is 1 if its argument is equal to x_i' and 0 otherwise.
% The new argument O is an adjacency matrix of observed edges, same shape as
% A, such that O(i,j)!=0 iff it's an observed edge (with O(i,j) being the
% observation value)

% (loopy) belief propagation for generic factor graph;
% custom psi (singleton potential) and phi (clique potential) can be
% defined for general problems; here we focus on the edge coloring problem
% return beliefs as a 3D tensor corresponding to the original adjmat A
% Yibo Yang, March 2018

function [beliefs, factor_graph] = cond_lbp2(A, O, w, its)
op = @sum;
% define potential functions for the coloring problem
phi = exp(w);
% clique potential for coloring problem; 1 if no color clash, 0 o/w
% psi = @(cfg) length(unique(cfg)) == length(cfg);

factor_graph = convert_to_factor_graph(A);
N = factor_graph.N;
E = factor_graph.E;  % num r.v.s
K = length(w);
neighbors=factor_graph.neighbors;
M = ones(N, N, K);  % central message storage; details see calc_message
for t=1:its
    for i=1:N
        for j=neighbors{i}
            phi_j = phi;
            if i>E && length(neighbors{i})==1    % if C_i is the singleton clique over var j
                edge_j =  factor_graph.orig_edges(j, :);
                obs = O(edge_j(1), edge_j(2));  % assuming the O matrix is symmetric
                if obs  % furthermore, if var j is observed
                    phi_j = zeros(1, K);
                    phi_j(obs) = phi(obs);  % multiply phi by evidence indicator
                end
            end
            msg= calc_message(factor_graph, phi_j, @psi, M, i, j, op);
            M(i,j,:) =msg;
        end
    end
end

% calc beliefs (approximate marginals)
% beliefs is a cell array of the same shape as the original adjmat A;
% beliefs{i,j} stores the belief (approximate marginal) of edge (i,j) as a
% K-vector; beliefs{c,c} stores the belief of clique c, where c corresponds
% to the vertex c in adjmat A that connects Nc edges together (we call such
% a vertex a "connecting vertex"), so that beliefs{c,c} is a vector of
% length K^Nc, with each entry corresponding to a clique configuration in
% sorted order (e.g., if K=3, Nc=2, then the entries are the belief values
% for configuration 11, 12, 13, 21, 22, 23, 31, 32, 33 (some of which will
% be zeros due to coloring constraint)). 
beliefs = cell(length(A));

% singleton beliefs; equivalent to the corresponding singleton clique marginals in clique_marg
E = factor_graph.E;
for e=1:E
    nbr_msg = ones(K, 1);
    for j=neighbors{e}  % mult all incoming messages (including from singleton cliques)
        nbr_msg = nbr_msg .* squeeze(M(j, e, :));
    end
    nbr_msg = nbr_msg/sum(nbr_msg);
    nbr_msg = nbr_msg';
    
    edge = factor_graph.orig_edges(e,:);
    beliefs{edge(1), edge(2)} = nbr_msg;
    beliefs{edge(2), edge(1)} = nbr_msg;  % nice & symmetric
end

num_nonsingleton_cliques = length(factor_graph.orig_connecting_verticies);
D = E + num_nonsingleton_cliques;  % D==N-E

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
    
    orig_v_id = factor_graph.orig_connecting_verticies(c-E);    % connecting vertex id in original adjmat A
    beliefs{orig_v_id, orig_v_id} = c_marg;
end

end



% helper functions
function good = psi(cfg)
    % clique potential for coloring problem; 1 if no color clash, 0 o/w
    good = length(unique(cfg)) == length(cfg);
end


