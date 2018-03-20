function main()
  data = load('data1.txt'); %Regularized Multi-variable Logistic Regression
  X = data(:, 1:(size(data,2) - 1));
  y = data(:, end);

  X = [ones(size(X, 1), 1) X]; %Concatenate 1s to X to include bias in matmul
  theta = zeros(size(X,2),1);
  lambda = 0; %Regularization parameter

  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [theta, cost, exitFlag] = fminunc(@(t)(computeCost(X, y, t, lambda)), theta, options);

  printf('Theta derived from fminunc:\n')
  printf('%f\n', theta)

  figure(1)
  hold on


  idxs = find(y == 0);
  plot(X(idxs, 2), X(idxs, 3), 'k+');
  idxs = find(y == 1);
  plot(X(idxs, 2), X(idxs, 3), 'bo');

  xBound = [min(X(:,2))-2,  max(X(:,2))+2]; %Plot boundary line
  yBound = (-1./theta(3)).*(theta(2).*xBound + theta(1));
  plot(xBound, yBound)
  axis([30, 100, 30, 100])
  xlabel('Exam 1 score')
  ylabel('Exam 2 score')
  legend('Admitted', 'Not admitted')

  printf('Training accuracy: %2f\n', mean(double(predict(X, theta) == y)))
