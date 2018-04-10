function Marg = gibbs(A, w, burnin, its)
    % estimate marginal probabilities of r.v.s
    % use gibbs_sample to generate samples for the coloring problem
    % returns the estimated marginal probabilities of edges of A as a ExExK
    % tensor Marg, such that Marg(i,j,k) is the probability that edge (i,j) is
    % colored with color k
    % Yibo Yang, March 2018
    
    factor_graph = convert_to_factor_graph(A);
    E = factor_graph.E; % num r.v.s in factor graph=num edges in original graph
    K = length(w);
    samples = gibbs_sample(A, w, burnin, its);
    S = size(samples, 1);
    Marg = zeros(length(A), length(A), K);
    for e=1:E
        edge = factor_graph.orig_edges(e, :);
        counts = hist(samples(:, e), 1:K);
        Marg(edge(1), edge(2), :) = counts / S;
        Marg(edge(2), edge(1), :) = counts / S;  % nice & symmetric
    end
end
