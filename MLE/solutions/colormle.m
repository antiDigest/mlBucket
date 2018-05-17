function w = colormle(A, samples)
%COLORMLE: MLE for learning parameters w of the coloring problem
%   A is the adjacency matrix of original graph; 
%   samples is a 3D tensor (see output of gibbs_sample2.m) such that each 
% slice, samples(:,:,t)(i,j), is the color of edge (i,j) in the original graph A
% Yibo Yang, April 2018

K = max(max(max(samples)));
S = size(samples, 3);   % num samples

% convert to matrix of samples, each row a sample
factor_graph = convert_to_factor_graph(A);
E = factor_graph.E; % num r.v.s in factor graph=num edges in original graph
mat_samples = zeros(S, E);
for e=1:E
    edge = factor_graph.orig_edges(e, :);
    mat_samples(:, e) = squeeze(samples(edge(1), edge(2), :));
end

% convert samples to empirical counts as a K-vector
all_samples = mat_samples(:);
counts = hist(all_samples, 1:K);
empirical_stats = counts / S;

% some run settings for MLE
step_size = 0.2;
its = 100;
eps = 1e-3;
init_w = zeros(1, K);    % init to zero

w = colormle_worker(A, empirical_stats, init_w, step_size, its, eps);
end

