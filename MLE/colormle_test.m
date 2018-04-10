clc
clear all

A = [0 1 1;
    1 0 0;
    1 0 0];

weight = [1 2 3];

samples = gibbs_sample2(A, weight, 1000, 100000);

w = colormle(A, samples);

[belief2, fg] = lbp([0 1 1; 1 0 0; 1 0 0], weight, 100);
[belief1, fg] = lbp([0 1 1; 1 0 0; 1 0 0], w, 100);

belief1{1}
belief2{1}