clear;
seed = 0;
rng(seed);

% simple chain graph A--B--C
A = [0 1 0; 1 0 1; 0 1 0];

% ground truth w
w = [-1 0 1];
samples = gibbs_sample2(A, w, 10000, 1000);
save('ps4_samples.mat', 'samples');
