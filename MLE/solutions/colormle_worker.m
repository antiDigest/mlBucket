function w = colormle_worker(A, empirical_stats, init_w, step_size, its, eps)
% perform MLE learning of singleton potential parameter w (shared by all
% r.v.s, phi(x) = exp(w_x))
% empirical_stats is a K-vector of empirical counts of r.v.s in each state
% its = max num iterations; terminates early if ||mle_gradient|| < eps
% Output w should approach true solution subtracted by a constant value;
% Yibo Yang, April 2018

bp_its = 3;
K = length(empirical_stats);
w = init_w;
for t=1:its
    [beliefs, factor_graph] = lbp2(A, w, bp_its);
    % extract the vector of expected suff stats from beliefs
    expected_stats = zeros(1, K);
    E = factor_graph.E;
    for e=1:E
        edge = factor_graph.orig_edges(e, :);
        expected_stats = expected_stats + beliefs{edge(1), edge(2)};
    end
%     disp(expected_stats);
    grad = empirical_stats - expected_stats;
    
    w = w + step_size * grad;   % gradient ascent
    if norm(grad) < eps
        disp('mle converged: norm(grad) < eps');
        break
    end

end

end

