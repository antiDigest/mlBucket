% Find an (approximate) MAP configuration by max-product and returning the
% MAP configurations of node marginals (could be inconsistent; correct way
% to do this is via backtracking)
% Yibo Yang, March 2018

function X = maxprod(A, w, its)
op = @max;
% define potential functions for the coloring problem
phi = exp(w);
% clique potential for coloring problem; 1 if no color clash, 0 o/w
% psi = @(cfg) length(unique(cfg)) == length(cfg);

factor_graph = convert_to_factor_graph(A);
N = factor_graph.N;
E = factor_graph.E;
K = length(w);
neighbors = factor_graph.neighbors;
M = ones(N, N, K);  % central message storage; details see calc_message
for t=1:its
    for i=1:N
        for j=neighbors{i}
            M(i,j,:) = calc_message(factor_graph, phi, @psi, M, i, j, op);
        end
    end
end

% calc marginal max-beliefs and MAP config
X = zeros(length(A));
tol = 1e-4;  % consider numbers x and y equal if abs(x-y) <= tol
for e=1:E   % node id counts from 1
    nbr_msg = ones(K, 1);
    for j=neighbors{e}  % mult all incoming messages (including from singleton cliques)
        nbr_msg = nbr_msg .* squeeze(M(j, e, :));
    end
    max_marg = max(nbr_msg);
    arg_max = find(abs(nbr_msg-max_marg)<=tol);
    if length(arg_max)==1  % if unique; otherwise set MAP config 0 by default
        edge = factor_graph.orig_edges(e, :);
        X(edge(1), edge(2)) = arg_max;
        X(edge(2), edge(1)) = arg_max;  % nice & symmetric
    end
end

end


function good = psi(cfg)
    % clique potential for coloring problem; 1 if no color clash, 0 o/w
    good = length(unique(cfg)) == length(cfg);
end