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