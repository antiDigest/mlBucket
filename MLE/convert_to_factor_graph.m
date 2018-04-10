function factor_graph = convert_to_factor_graph(original_adjmat)
    %% Return a struct 'factor_graph' representing the factor graph based on
    % the original graph's adjacency matrix
    % Antriksh Agarwal, April 2018
    
    % find edges in original graph
    upper_triu = triu(original_adjmat);
    [row, col] = find(upper_triu);
    edges = [row, col];    % Ex2 mat, each row gives adjacent node ids (i,j) in original graph
    num_edges = size(edges,1);
    E = num_edges;
    
    % any vertex in the original graph with degree>=2 makes a factor
    degrees = sum(original_adjmat, 1);
    connecting_vertices = find(degrees>1);
    num_nonsingleton_factors = length(connecting_vertices);
    % each node (edge in the original graph) also gets a singleton factor
    num_cliques = num_nonsingleton_factors + num_edges;
    C = num_cliques;
    N = num_edges + num_cliques;
    adjmat = zeros(N);
    % connect adjacent edges (with connecting vertices, i.e. nonsingleton factors)
    for e=1:num_edges
        edge = edges(e, :);
        for v=edge
            cv_idx = find(v==connecting_vertices);
            if cv_idx  % if an adjacent vertex is a connecting vertex
                clique_id = num_edges + cv_idx; % upper-left ExE submatrix for r.v.s
                adjmat(e, clique_id) = 1;
            end
        end
    end
    for e=1:num_edges
        clique_id = num_edges+num_nonsingleton_factors+e; % singleton clique
        adjmat(e, clique_id) = 1;
    end
    adjmat = triu(adjmat)+triu(adjmat,1)';
    
    % create a lookup table for finding neighbors in factor graph
    neighbors = cell(N, 1);
    for n=1:N
        neighbors{n} = find(adjmat(n,:));
    end
    
    % below two fields refer to the original graph, in order to refer/convert back
    factor_graph.orig_edges = edges;  % mapped to factor nodes 1:E
    factor_graph.orig_connecting_verticies = connecting_vertices;  % mapped to nonsingleton factor cliques
    % below refer to the factor graph
    factor_graph.adjmat = adjmat;
    factor_graph.E = E;
    factor_graph.C = C;
    factor_graph.N = N;
    factor_graph.neighbors = neighbors;
    
    
end

