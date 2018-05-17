function samples = gibbs_sample2(A, w, burnin, its)
%GIBBS_SAMPLE Generate samples for the coloring problem.
% call gibbs_sample and convert samples into length(A) by length(A) by its
% tensor, such that each slice samples(:,:,t)(i,j) is the color of edge
% (i,j) in the original graph A
% This is to make the output independent of the edge ordering (in
% factor_graph representation), to make PS4 easier.
% Yibo Yang, March 2018

samples_ = gibbs_sample(A, w, burnin, its);
factor_graph = convert_to_factor_graph(A);
assert(its==length(samples_), 'num samples should equal its');

samples = zeros(length(A), length(A), its);
for e=1:factor_graph.E
    edge=factor_graph.orig_edges(e, :);
    samples(edge(1), edge(2), :) = samples_(:, e);
    samples(edge(2), edge(1), :) = samples_(:, e);  % nice & symmetric
end
end



function samples = gibbs_sample(A, w, burnin, its)
%GIBBS_SAMPLE Generate samples for the coloring problem.
% samples is an its by E matrix, where E is the number of edges in A;
% the columns are ordered by factor_graph.orig_edges (see
% convert_to_factor_graph for details)
% Yibo Yang, March 2018

factor_graph = convert_to_factor_graph(A);
N = factor_graph.N; % num all nodes+factors in factor_graph
E = factor_graph.E; % num r.v.s in factor graph=num edges in original graph
K = length(w);
% define potential functions for the coloring problem
phi = exp(w);
    function good = psi(cfg)
        % clique potential for coloring problem; 1 if no color clash, 0 o/w
        good = length(unique(cfg)) == length(cfg);
    end

neighbors=factor_graph.neighbors;


% init_state; find a valid initial coloring assignment
% first convert factor_graph representation to adjacency matrix over the
% random variables (i.e., edges in the original graph)
edge_adjmat = zeros(E);
for cv_id=(E+1):(N-E)   % loop over connecting_vertices
    connected_edges = neighbors{cv_id};
    connections = combnk(connected_edges, 2);
    for i=1:size(connections,1)  % connect all adjacent edges (indicent to cv)
        con = connections(i, :);
        edge_adjmat(con(1), con(2)) = 1;
    end
end
edge_adjmat = triu(edge_adjmat)+triu(edge_adjmat,1)';

% get initial coloring
[state, K_init] = greedy_color(edge_adjmat);
% state is the vector of states of all r.v.s
assert(K_init <= K, 'unable to produce a greedy initial coloring (not enough colors?)');

% get samples
samples = zeros(its, E);    % its is # of samples after burnin
for t=1:(burnin+its)
    for e=1:E
        nbrs = neighbors{e};    % the ids of all the factors that e is involved in
        likelihood = zeros(1, K);     % conditional probability to sample new Xe from
        for k=1:K    % set xe to every color
            prod=1;
            % multiply by every neighboring factor (making up e's markov
            % blanket)
            for nbr_id=nbrs
                factor_vars = neighbors{nbr_id};   % factor scope
                factor_size = length(factor_vars);
                if factor_size == 1
                    assert(factor_vars==e);
                    % this is a singleton factor over e only
                    prod = prod*phi(k);
                else
                    factor_cfg = zeros(1, factor_size);
                    for i=1:factor_size
                        v = factor_vars(i);
                        if v==e  % current node
                            factor_cfg(i) = k;
                        else
                            factor_cfg(i) = state(v);
                        end
                    end
                    prod = prod*psi(factor_cfg);
                end
            end
            likelihood(k) = prod;
        end
        prob = likelihood / sum(likelihood);
        
        % https://stackoverflow.com/a/13914141
        x = sum(rand >= cumsum([0, prob]));
        state(e) = x;
    end
    
    if t>burnin
        samples(t-burnin, :) = state;
    end
end

end

function [coloring, K] = greedy_color(adjmat)
% Try to color a graph using the smallest number of colors with a greedy strategy; return a valid coloring
% and the number of colors used;
% rv_adjmat is the adjacency matrix of nodes (r.v.s)
N = length(adjmat);
neighbors = cell(N, 1);
for n=1:N
    neighbors{n} = find(adjmat(n,:));
end
coloring = ones(1, N) * -1; % colors of all nodes in a vector
K = 1;  % this is our budget, current number of colors used
for n=1:N
    colors_used = zeros(1, K);  % bool array indicating which colors are used by neighbors
    nbrs = neighbors{n};
    nbr_colors = zeros(1, length(nbrs));
    for b=1:length(nbrs)
        nbr_colors(b) = coloring(nbrs(b));
    end
    for c=nbr_colors
        if c~=-1
            colors_used(c) = 1;
        end
    end
    unused_color = find(~colors_used);
    if unused_color
        coloring(n) = unused_color(1);  % assign the first unused color
    else    % need a new color
        K = K + 1;
        coloring(n) = K;
    end
end
end