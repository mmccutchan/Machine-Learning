function pred = predict(X, theta)

  pred = sigmoid(X * theta);
  for i = 1:length(X)
    if pred(i) >= 0.5
      pred(i) = 1;
    else
      pred(i) = 0;
    end
  end
