clear;
load('ps4_samples.mat');

% simple chain graph A--B--C
A = [0 1 0; 1 0 1; 0 1 0];
% ground truth w
w = [-1 0 1];

% mle
w_mle = colormle(A, samples);
disp('learned w_mle=');
disp(w_mle);

% em
try
	L = [0 0 0; 0 0 1; 0 1 0];
	% some people's code required 'L = logical([0 1 0; 1 0 0; 0 0 0]);' if the above crashes
	w_em = colorem(A, L, samples);
catch
	L = logical([0 1 0; 1 0 0; 0 0 0]);
	% some people's code required this to run
	w_em = colorem(A, L, samples);
end
disp('learned w_em=');
disp(w_em);
