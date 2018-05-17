function prev_w = colorem(A, L, samples)
%COLOREM: EM for learning parameters w of the coloring problem
%   A is the adjacency matrix of original graph;
%   L is the adjacency matrix of missing edges, same shape as A;
%   samples is a 3D tensor (see output of gibbs_sample2.m) such that each 
% slice, samples(:,:,t)(i,j), is the color of edge (i,j) in the original graph A
% Yibo Yang, April 2018

K = max(max(max(samples)));
S = size(samples, 3);   % num samples

L = triu(L)+triu(L,1)';
not_L = ~L; % adjacency mat of observed edges

% convert samples to partial observations by keeping only observed edges
observed = zeros(size(samples));
for s=1:S
    observed(:,:,s) = samples(:,:,s) .* not_L;
end


prev_w = zeros(1, K);    % init to zero

% EM settings
em_its = 20;
eps = 1e-3;  % crude convergence criterion based on absolute norm
% E-step settings
bp_its = 3;
% M-step settings
step_size = 0.2;
m_its = 50;

for t=1:em_its
    % E-step
    stats = zeros(1, K);   % expected/empirical sufficient stats
    for s=1:S
        O = observed(:, :, s);
        [beliefs, factor_graph] = cond_lbp2(A, O, prev_w, bp_its);
        % extract the vector of expected suff stats from beliefs
        E = factor_graph.E;
        for e=1:E
            edge = factor_graph.orig_edges(e, :);
            stats = stats + beliefs{edge(1), edge(2)};
        end
    end
    stats = stats / S;
    disp('e_step stats:');
    disp(stats);
    disp('w');
    disp(prev_w);
    
    % M-step
    w = colormle_worker(A, stats, prev_w, step_size, m_its, eps);
    if norm(w - prev_w) < eps
        disp('em converged');
        break;
    end
    prev_w = w;
    
end
end