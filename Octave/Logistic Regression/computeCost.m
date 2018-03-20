function [cost, gradient] = computeCost(X, y, theta, lambda)
  m = length(y);
  cost = 1 / m .* (sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + lambda / 2 .* sum(theta(2:length(theta)) .^ 2));

  gradient = 1 / m * sum((sigmoid(X * theta) - y) .* X) + (lambda / m * theta)';
  gradient(1) = (1 / m * sum(sigmoid(X * theta) - y) * X)(1);
