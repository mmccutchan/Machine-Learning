function [theta, costs] = gradientDescent(X, y, theta, learningRate = 0.01, epochs = 100)

m = length(y);
costs = zeros(epochs);

for i = 1:epochs
  theta = theta - (learningRate / m * sum((X * theta - y) .* X))';
  costs(i) = computeCost(X, y, theta);
end
