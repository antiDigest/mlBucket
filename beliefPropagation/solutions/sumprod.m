% sum-product for the coloring problem; convert to factor graph whose nodes
% are edges in the original graph and do message-passing in the factor
% graph (in each iteration go through every node and send messages to its
% neighbors); exact for tree-structured factor graphs, approximate for
% loopy ones; return Bethe free energy
% Yibo Yang, March 2018

function Z = sumprod(A, w, its)
    [beliefs, factor_graph] = lbp(A, w, its)
    % factor_graph = convert_to_factor_graph(A);
    N = factor_graph.N;
    E = factor_graph.E;
    K = length(w);
    neighbors = factor_graph.neighbors;
    
    % calc Bethe free energy
    % inner product term
    % only singleton cliques contribute, because 0 log0 = 0
    w_col = w(:);   % canonical params (theta)
    node_marg = cell2mat(beliefs(1:E));
    disp(node_marg)
    disp(w_col)
    inner_prod = sum(node_marg * w_col); % matrix times vector here
    % entropy:
    % singleton entropy term
    H = - sum(sum(node_marg .* log(node_marg)));
    % mutual information term
    for c=E+1:size(beliefs,1)   % go thru non-singleton cliques only;
        %(factor_graph.C-factor_graph.E)=N-2E=size(beliefs,1) of them
        nbrs = neighbors{c};
        clique_size=length(nbrs);
        assert(clique_size>1, ['clique with id ' num2str(c) ' cannot be singleton!']);
        % skip singleton cliques because we've already computed singleton
        % entropy above
        clique_config = permn(1:K, clique_size);
        num_config = length(clique_config);
        for n=1:num_config
            cfg = clique_config(n, :);
            prod = psi(cfg);
            if prod  % else, invalid clique config, 0log0=0, skip
                for nbr_idx=1:clique_size
                    nbr_id = nbrs(nbr_idx);
                    nbr_cfg = cfg(nbr_idx);
                    prod = prod * node_marg(nbr_id, nbr_cfg); % should be the same as using beliefs{nbr_id}(nbr_cfg)
                end
                H = H - beliefs{c}(n) * log(beliefs{c}(n) / prod);
            end
        end
    end
    
    Z = exp(inner_prod + H);
end



function good = psi(cfg)
    % clique potential for coloring problem; 1 if no color clash, 0 o/w
    good = length(unique(cfg)) == length(cfg);
end