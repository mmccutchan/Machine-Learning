function main()
  data = load('data1.txt'); %Multi-variable Linear Regression
  X = data(:, 1:(size(data,2) - 1));
  y = data(:, end);
  m = length(y);
  
  X = normalizeData(X);
 
  X = [ones(size(X, 1), 1) X]; %Concatenate 1s to X to include bias in matmul
  theta = zeros(size(X,2),1);
  
  [theta, costs] = gradientDescent(X, y, theta, 0.01, 150); %Train parameters
  figure(1)

  plot(costs)
  title('Cost at Iteration')
  xlabel('Iteration')
  ylabel('Cost')

  printf('Theta derived from gradient descent:\n')
  printf('%f\n', theta)

  theta = normalEquation(X, y);
  printf('Theta derived from normal equation:\n')
  printf('%f\n', theta)

  figure(2)
  hold on
  plot(X(:,2:end), y, 'rx')
  plot(X(:,2:end), X * theta, '-')
  xlabel('Iterations')
  ylabel('Cost')
  title('Cost over Iterations')
