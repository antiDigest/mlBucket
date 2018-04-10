
% @author: Antriksh Agarwal
% Version 0: 04/05/2018

function w = colorem(A, L, samples)
    
    COLORS = max(samples(:)); % max color
    w = rand(1, COLORS); % initial random weights
    M = size(samples, 3); % Number of samples
    N = size(A, 2); % Number of nodes
    alpha = 0.1; % Learning Rate, step size
    % define potential functions for the coloring problem
    phi = exp(w);
    function good = psi(cfg)
        % clique potential for coloring problem; 1 if no color clash, 0 o/w
        good = length(unique(cfg)) == length(cfg);
    end
    
    factor_graph = convert_to_factor_graph(A);
    E = factor_graph.E;
    latent = [];
    observed = [];
    for e = 1:E
        edge = factor_graph.orig_edges(e, :);
        if (L(edge(1), edge(2)) ~= 1) % to skip those which are latent
            observed = [observed e];
            continue;
        end
        latent = [latent e];
    end
    %     latent
    %     observed
    %     return;
    
    C = factor_graph.C;
    
    Q = zeros(E, COLORS);
    P = Q;
    theta = w;
    
    % Calculating SUM(ij)SUM(m) (FEATURE-MAP(xij))
    emp_prob = zeros(E, COLORS); % FEATURE MAP for all edges - Emperical Probability
    for e = 1:E % Loop over all nodes in factor graph (edges in real graph)
        edge = factor_graph.orig_edges(e, :);
        
        f = zeros(M, COLORS);
        for m = 1:M % loop over all samples
            color = samples(edge(1), edge(2), m);
            f(m, color) = 1;
        end
        emp_prob(e, :) = sum(f, 1);
    end
    
    for its = 1:100
        %     while sum((emp_prob / M) - infer_prob) > 0.00001 % loop until the moment matching condition is satisfied
        
        [belief, fg] = lbp(A, w, 11);
        theta = w;
        phi = exp(w);
        prob = zeros(E, COLORS);
        for m = 1:M
            for e = latent % P(xe1, xe2, xe3=? | xe1, xe2)
                likelihood = zeros(1, COLORS);     % conditional probability
                for k = 1:COLORS % set xe to every color
                    prod = phi(k);
                    % take into account every observed factor
                    for obs = observed
                        nbrs = factor_graph.neighbors{obs};
                        for nbr_id=nbrs
                            factor_vars = factor_graph.neighbors{nbr_id};   % factor scope
                            factor_size = length(factor_vars);
                            if factor_size == 1 % Calculating PHI
                                assert(factor_vars == obs);
                                % this is a singleton factor over e only
                                edge = factor_graph.orig_edges(obs, :);
                                color = samples(edge(1), edge(2), m);
                                prod = prod * phi(color);
                            else % Calculating PSI
                                factor_cfg = zeros(1, factor_size);
                                for i=1:factor_size
                                    v = factor_vars(i);
                                    if v == e  % current node
                                        factor_cfg(i) = k;
                                    else
                                        edge = factor_graph.orig_edges(v, :);
                                        color = samples(edge(1), edge(2), m);
                                        factor_cfg(i) = color;
                                    end
                                end
                                prod = prod * psi(factor_cfg); % Calculating PRODUCT of PSI and PHI
                            end
                        end
                        likelihood(k) = prod; % Calculating likelihood of k being the missing value
                    end
                end
                prob(e, :) = prob(e, :) + likelihood / sum(likelihood);
            end
        end
        
        latent_prob = prob / M;
        
        % Calculating SUM(m)SUM(ij) {SUM(xij) (PROB(xij | w))}
        for e = 1:E
            infer_prob(e, :) = belief{e};
        end
        
        J = (latent_prob .* emp_prob - (infer_prob .* emp_prob) .* latent_prob); % Gradient
        J = J ./ exp(theta);
        theta = log(exp(theta) + (alpha * J)); % Gradient Ascent
        
        w = mean(theta, 1)
        
        alpha = 2 / (2 + its);
%         its = its + 1;
    end
    
end
