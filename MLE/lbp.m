% (loopy) belief propagation for generic factor graph;
% custom psi (singleton potential) and phi (clique potential) can be
% defined for general problems; here we focus on the edge coloring problem
% Yibo Yang, March 2018

function [beliefs, factor_graph] = lbp(A, w, its)
    op = @sum;
    % define potential functions for the coloring problem
    % phi = exp(w);
    % clique potential for coloring problem; 1 if no color clash, 0 o/w
    % psi = @(cfg) length(unique(cfg)) == length(cfg);
    
    factor_graph = convert_to_factor_graph(A);
    N = factor_graph.N;
    K = length(w);
    neighbors=factor_graph.neighbors;
    M = ones(N, N, K);  % central message storage; details see calc_message
    for t=1:its
        for i=1:N
            for j=neighbors{i}
                M(i,j,:) = calc_message(factor_graph, w, M, i, j, op);
            end
        end
    end
    
    % calc beliefs (approximate marginals)
    
    % singleton beliefs; equivalent to the corresponding singleton clique marginals in clique_marg
    E = factor_graph.E;
    node_marg = ones(E, K);
    for e=1:E   % node id counts from 1
        nbr_msg = ones(K, 1);
        for j=neighbors{e}
            nbr_msg = nbr_msg .* squeeze(M(j, e, :));
        end
        nbr_msg = nbr_msg/sum(nbr_msg);
        node_marg(e,:) = nbr_msg';
    end
    
    num_nonsingleton_cliques = length(factor_graph.orig_connecting_verticies);
    D = E + num_nonsingleton_cliques;  % D==N-E
    % beliefs is a Dx1 cell array; the first E entries store node marginals
    % (same as singleton clique marginals) in the order of
    % factor_graph.orig_edges and the rest num_nonsingleton_cliques entries
    % store (non-singleton) clique marginals in the order of
    % factor_graph.orig_connecting_verticies;
    % the node marginal correspond to [P(x=1); P(x=2); ... P(x=k)];
    % the clique marginal correspond to (say clique_size=2, K=3):
    % [P(x1=1,x2=1); P(x1=1,x2=2); P(x1=1,x2=3); P(x1=2,x2=1); P(x1=2,x2=2);...
    % P(x1=3,x2=3)], i.e., each clique marginal is a K^clique_size vector whose
    % elements correspond to joint clique configurations in sorted order
    
    beliefs = cell(D, 1);
    for e=1:E
        beliefs{e} = node_marg(e,:);  % copying over
    end
    
    % clique beliefs
    % to keep things simple here we only compute nonsingleton clique marginals
    % (the singleton clique marginals are the same as node marginals)
    for c=(E+1):D   % nonsingleton clique id
        nbrs = neighbors{c};
        clique_size=length(nbrs);
        assert(clique_size>1, 'should be a nonsingleton clique!');
        clique_config = permn(1:K, clique_size); % 111, 112, 113, 121, 122, 123, ...
        num_config = length(clique_config);
        c_marg = zeros(num_config, 1);
        for n=1:num_config
            cfg = clique_config(n, :);
            prod = psi(cfg);    % multiply by clique potential
            if prod==1
                for nbr_idx=1:clique_size
                    nbr_id = nbrs(nbr_idx);
                    nbr_cfg = cfg(nbr_idx);
                    prod = prod * M(nbr_id, c, nbr_cfg);
                end
            end
            c_marg(n) = prod;
        end
        c_marg = c_marg / sum(c_marg);
        beliefs{c} = c_marg;
    end
    
end



% helper functions
function good = psi(cfg)
    % clique potential for coloring problem; 1 if no color clash, 0 o/w
    good = length(unique(cfg)) == length(cfg);
end

function msg = calc_message(factor_graph, w, M, j, i, op)
    % calculate the message m_ji, from j to i, using the central storage M of messages
    % calculates either sum-product message (op=sum) or max-product message(op=max)
    % we also normalize each message to avoid overflow/underflow
    % phi, psi are callables; phi is the singleton potential; psi is the clique potential
    % in the coloring problem, phi = exp(w), psi = 1 iff all of its args are
    % unique, else 0;
    % M is a NxNxK tensor, such that the (i,j,:)th entry is a vector that store the values
    % of the function m_ij(x_j) for all the K states of x_j;
    % return a length K vector
    
    
    % setup
    K = size(M, 3); %=length(w)
    E = factor_graph.E; % num nodes
    % define potential functions for the coloring problem
    phi = exp(w);
    
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




function [M, I] = permn(V, N, K)
    % from https://www.mathworks.com/matlabcentral/fileexchange/7147-permn-v--n--k-
    % PERMN - permutations with repetition
    %   Using two input variables V and N, M = PERMN(V,N) returns all
    %   permutations of N elements taken from the vector V, with repetitions.
    %   V can be any type of array (numbers, cells etc.) and M will be of the
    %   same type as V.  If V is empty or N is 0, M will be empty.  M has the
    %   size numel(V).^N-by-N.
    %
    %   When only a subset of these permutations is needed, you can call PERMN
    %   with 3 input variables: M = PERMN(V,N,K) returns only the K-ths
    %   permutations.  The output is the same as M = PERMN(V,N) ; M = M(K,:),
    %   but it avoids memory issues that may occur when there are too many
    %   combinations.  This is particulary useful when you only need a few
    %   permutations at a given time. If V or K is empty, or N is zero, M will
    %   be empty. M has the size numel(K)-by-N.
    %
    %   [M, I] = PERMN(...) also returns an index matrix I so that M = V(I).
    %
    %   Examples:
    %     M = permn([1 2 3],2) % returns the 9-by-2 matrix:
    %              1     1
    %              1     2
    %              1     3
    %              2     1
    %              2     2
    %              2     3
    %              3     1
    %              3     2
    %              3     3
    %
    %     M = permn([99 7],4) % returns the 16-by-4 matrix:
    %              99     99    99    99
    %              99     99    99     7
    %              99     99     7    99
    %              99     99     7     7
    %              ...
    %               7      7     7    99
    %               7      7     7     7
    %
    %     M = permn({'hello!' 1:3},2) % returns the 4-by-2 cell array
    %             'hello!'        'hello!'
    %             'hello!'        [1x3 double]
    %             [1x3 double]    'hello!'
    %             [1x3 double]    [1x3 double]
    %
    %     V = 11:15, N = 3, K = [2 124 21 99]
    %     M = permn(V, N, K) % returns the 4-by-3 matrix:
    %     %        11  11  12
    %     %        15  15  14
    %     %        11  15  11
    %     %        14  15  14
    %     % which are the 2nd, 124th, 21st and 99th permutations
    %     % Check with PERMN using two inputs
    %     M2 = permn(V,N) ; isequal(M2(K,:),M)
    %     % Note that M2 is a 125-by-3 matrix
    %
    %     % PERMN can be used generate a binary table, as in
    %     B = permn([0 1],5)
    %
    %   NB Matrix sizes increases exponentially at rate (n^N)*N.
    %
    %   See also PERMS, NCHOOSEK
    %            ALLCOMB, PERMPOS on the File Exchange
    
    % tested in Matlab 2016a
    % version 6.1 (may 2016)
    % (c) Jos van der Geest
    % Matlab File Exchange Author ID: 10584
    % email: samelinoa@gmail.com
    
    % History
    % 1.1 updated help text
    % 2.0 new faster algorithm
    % 3.0 (aug 2006) implemented very fast algorithm
    % 3.1 (may 2007) Improved algorithm Roger Stafford pointed out that for some values, the floor
    %   operation on floating points, according to the IEEE 754 standard, could return
    %   erroneous values. His excellent solution was to add (1/2) to the values
    %   of A.
    % 3.2 (may 2007) changed help and error messages slightly
    % 4.0 (may 2008) again a faster implementation, based on ALLCOMB, suggested on the
    %   newsgroup comp.soft-sys.matlab on May 7th 2008 by "Helper". It was
    %   pointed out that COMBN(V,N) equals ALLCOMB(V,V,V...) (V repeated N
    %   times), ALLCMOB being faster. Actually version 4 is an improvement
    %   over version 1 ...
    % 4.1 (jan 2010) removed call to FLIPLR, using refered indexing N:-1:1
    %   (is faster, suggestion of Jan Simon, jan 2010), removed REPMAT, and
    %   let NDGRID handle this
    % 4.2 (apr 2011) corrrectly return a column vector for N = 1 (error pointed
    %    out by Wilson).
    % 4.3 (apr 2013) make a reference to COMBNSUB
    % 5.0 (may 2015) NAME CHANGED (COMBN -> PERMN) and updated description,
    %   following comment by Stephen Obeldick that this function is misnamed
    %   as it produces permutations with repetitions rather then combinations.
    % 5.1 (may 2015) always calculate M via indices
    % 6.0 (may 2015) merged the functionaly of permnsub (aka combnsub) and this
    %   function
    % 6.1 (may 2016) fixed spelling errors
    
    narginchk(2,3) ;
    
    if fix(N) ~= N || N < 0 || numel(N) ~= 1 ;
        error('permn:negativeN','Second argument should be a positive integer') ;
    end
    nV = numel(V) ;
    
    if nargin==2, % PERMN(V,N) - return all permutations
        
        if nV==0 || N == 0,
            M = zeros(nV,N) ;
            I = zeros(nV,N) ;
            
        elseif N == 1,
            % return column vectors
            M = V(:) ;
            I = (1:nV).' ;
        else
            % this is faster than the math trick used for the call with three
            % arguments.
            [Y{N:-1:1}] = ndgrid(1:nV) ;
            I = reshape(cat(N+1,Y{:}),[],N) ;
            M = V(I) ;
        end
    else % PERMN(V,N,K) - return a subset of all permutations
        nK = numel(K) ;
        if nV == 0 || N == 0 || nK == 0
            M = zeros(numel(K), N) ;
            I = zeros(numel(K), N) ;
        elseif nK < 1 || any(K<1) || any(K ~= fix(K))
            error('permn:InvalidIndex','Third argument should contain positive integers.') ;
        else
            
            V = reshape(V,1,[]) ; % v1.1 make input a row vector
            nV = numel(V) ;
            Npos = nV^N ;
            if any(K > Npos)
                warning('permn:IndexOverflow', ...
                    'Values of K exceeding the total number of combinations are saturated.')
                K = min(K, Npos) ;
            end
            
            % The engine is based on version 3.2 with the correction
            % suggested by Roger Stafford. This approach uses a single matrix
            % multiplication.
            B = nV.^(1-N:0) ;
            I = ((K(:)-.5) * B) ; % matrix multiplication
            I = rem(floor(I),nV) + 1 ;
            M = V(I) ;
        end
    end
    
    % Algorithm using for-loops
    % which can be implemented in C or VB
    %
    % nv = length(V) ;
    % C = zeros(nv^N,N) ; % declaration
    % for ii=1:N,
    %     cc = 1 ;
    %     for jj=1:(nv^(ii-1)),
    %         for kk=1:nv,
    %             for mm=1:(nv^(N-ii)),
    %                 C(cc,ii) = V(kk) ;
    %                 cc = cc + 1 ;
    %             end
    %         end
    %     end
    % end
end
