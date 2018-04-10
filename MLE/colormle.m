
% @author: Antriksh Agarwal
% Version 0: 04/02/2018

function w = colormle(A, samples)
    
    k = max(samples(:)); % max color
    w = ones(1, k); % initial random weights
    M = size(samples, 3); % Number of samples
    N = size(A, 2); % Number of nodes
    alpha = 0.1; % Learning Rate, step size
    
    factor_graph = convert_to_factor_graph(A);
    E = factor_graph.E;
        
    % Calculating SUM(ij)SUM(m) (FEATURE-MAP(xij))
    emp_prob = zeros(E, k); % FEATURE MAP for all edges - Emperical Probability
    for e = 1:E % Loop over all nodes in factor graph (edges in real graph)
        edge = factor_graph.orig_edges(e, :);
        
        f = zeros(M, k);
        for m = 1:M % loop over all samples
            color = samples(edge(1), edge(2), m);
            f(m, color) = 1;
        end
        emp_prob(e, :) = sum(f, 1);
    end
    
    infer_prob = zeros(E, k); % Inference Probability - SUM(m)SUM(ij) {SUM(xij) (PROB(xij | w))} * emp_prob
    theta = w; % Weights
    
    its = 0;
%     for its = 1:1000 % Loop for 1000 iterations
    while sum((emp_prob / M) - infer_prob) > 0.00001 % loop until the moment matching condition is satisfied
        
        % Calculating SUM(m)SUM(ij) {SUM(xij) (PROB(xij | w))}
        [belief, fg] = lbp(A, w, 11);
        for e = 1:E
            infer_prob(e, :) = belief{e};
        end
        
        J = (emp_prob - infer_prob .* emp_prob); % Gradient
        J = J ./ exp(theta);
        theta = log(exp(theta) + (alpha * J)); % Gradient Ascent
        
        w = mean(theta, 1);
        
        alpha = 2 / (2 + its);
        its = its + 1;
    end
    
end