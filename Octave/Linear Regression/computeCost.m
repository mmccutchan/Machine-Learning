function [cost] = computeCost(X, y, theta)
  cost = sum((X * theta - y) .^ 2) / 2 / length(y);
