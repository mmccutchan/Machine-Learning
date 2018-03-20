function [sigmoid] = sigmoid(X)
  ones = ones(size(X));
  e = ones * e;
  sigmoid = ones ./ (ones + e .^ -X);
