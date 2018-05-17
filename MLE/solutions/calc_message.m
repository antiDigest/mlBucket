function msg = calc_message(factor_graph, phi, psi, M, j, i, op)
% calculate the message m_ji, from j to i, using the central storage M of messages
% calculates either sum-product message (op=sum) or max-product message(op=max)
% we also normalize each message to avoid overflow/underflow
% phi, psi are callables; phi is the singleton potential; psi is the clique potential
% in the coloring problem, phi = exp(w), psi = 1 iff all of its args are
% unique, else 0;
% M is a NxNxK tensor, such that the (i,j,:)th entry is a vector that store the values
% of the function m_ij(x_j) for all the K states of x_j;
% return a length K vector
% Yibo Yang, March 2018

% setup
K = size(M, 3); %=length(w)
E = factor_graph.E; % num nodes
msg = zeros(K, 1); % represents fxn of xi

% check if j, i are cliques or nodes in the factor graph
if j<=E  % msg from node j to clique i; just a product of messages into j
    nbrs = factor_graph.neighbors{j};
    for k=1:K
        prod = 1;
        for nbr_id=nbrs
            if nbr_id==i
                continue
            else
                prod = prod * M(nbr_id, j, k);
            end
        end
        msg(k) = prod;
    end
else
    % msg from clique j to node i
    nbrs = factor_graph.neighbors{j};
    clique_size = length(nbrs);
    if clique_size==1  % singleton clique; sum over empty set, trivial
        msg = phi;  % msg is just singleton potential
    else
        clique_config = permn(1:K, clique_size);    % 111, 112, 113, 121, 122, 123, ...
        % go through every clique configuration
        num_config = length(clique_config); % K^clique_size
        nbr_msg_prod = zeros(num_config, 1);
        i_cfg = clique_config(:, 1);    % all xi configurations; let i be 1st position in clique
        nbrs_except_i = nbrs(nbrs~=i);
        for n=1:num_config
            cfg = clique_config(n, :);
            prod = psi(cfg);    % multiply by clique potential
            if prod==1
                for nbr_idx=1:(clique_size-1)
                    nbr_id = nbrs_except_i(nbr_idx);
                    nbr_cfg = cfg(nbr_idx+1);
                    prod = prod * M(nbr_id, j, nbr_cfg);
                end
            end
            nbr_msg_prod(n) = prod;
        end
        % https://www.mathworks.com/help/matlab/ref/accumarray.html (think
        % k-means m-step, sum/avg/max by clusters)
        % sum or max out
        msg = accumarray(i_cfg, nbr_msg_prod, [], op);
    end
end
msg = msg / sum(msg);
end


