clear;

Gs={};    % cell array to hold graphs for MRFs
% Zs, Xs contains correct answers for the tests on sumprod, maxprod
Zs = [];
Xs = {};

w1 = [0 0 0];   % to be used with sumprod; this corresponds to uniform distribution 
% on valid coloring and the answer should be the (approximate) number of valid coloring
w2 = [-1 0 1];  % to be used with maxprod; the MAP assignment should prefer color 3 while remaining valid

% trees
%1
% A--B
Gs{end+1}=[0 1; 1 0];
Zs(end+1)=3;
Xs{end+1}=[0 3; 3 0];
%2
%    D
%  / | \
% A  B  C
Gs{end+1}=[0 0 0 1; 0 0 0 1; 0 0 0 1; 1 1 1 0];
Zs(end+1)=6;
Xs{end+1}=[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];     % non-unique
%3
%    D
%  / | \
% A  B  C--E
Gs{end+1}= [0 0 0 1 0; 0 0 0 1 0; 0 0 0 1 1; 1 1 1 0 0; 0 0 1 0 0];
Zs(end+1)=12;
Xs{end+1}=[0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 3; 0 0 0 0 0; 0 0 3 0 0];

% loopys
%4
%    A
%   /  \
%   B--C
Gs{end+1}=[0 1 1; 1 0 1; 1 1 0];
Zs(end+1)=8;
Xs{end+1}=[0 0 0; 0 0 0; 0 0 0];     % non-unique
%5
%    A
%  / | \
% B  |  C
%  \ | /
%    D
Gs{end+1}=[0 1 1 1; 1 0 0 1; 1 0 0 1; 1 1 1 0];
Zs(end+1)=16/3;
Xs{end+1}=[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0];     % non-unique


its = 100;
sp_right = 0;
sp_tol = 1e-2;  % ZB should be 1% within ground truth
sp_failed = [];

mp_right = 0;
mp_tol = 0.1;   % 90% of the MAP assignment entries should agree with ground truth
mp_failed = [];

for t=1:length(Gs)
    G=Gs{t};
    Z=Zs(t);
    X=Xs{t};
    
    try
        Zt=sumprod(G, w1, its);
    catch
        Zt = 0;
    end
    
    if abs(Z-Zt)/Zt <= sp_tol
        sp_right=sp_right+1;
    else
        sp_failed(end+1)=t;
        disp(['got Zt', num2str(t), '=', num2str(Zt)]);
    end
    
    try
        Xt=maxprod(G, w2, its);
    catch
        Xt=zeros(10);  % cannot be correct
    end
    % either upper or lower triangular of maxprod answer needs to be correct
    N = size(X, 1);
    trian_size = (1+N)*N/2;
    if all(size(Xt)==size(X)) && (sum(sum(triu(X)~=triu(Xt)))/trian_size <= mp_tol ...
            || sum(sum(tril(X)~=tril(Xt)))/trian_size <= mp_tol)
        mp_right=mp_right+1;
    else
        mp_failed(end+1)=t;
    end
end

% https://stackoverflow.com/questions/14924181/how-to-display-print-vector-in-matlab
fprintf('sumprod failed to pass tests on graph %s\n', strjoin(cellstr(num2str(sp_failed(:))),', '));
fprintf('maxprod failed to pass tests on graph %s\n', strjoin(cellstr(num2str(mp_failed(:))),', '));
disp(['sp_right=', num2str(sp_right)]);
disp(['mp_right=', num2str(mp_right)]);
