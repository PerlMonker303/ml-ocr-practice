function [J, grad] = computeCost (nn_params, input_layer, hidden_layer, output_layer, X_train, y_train, lambda)
  # Reshaping parameters
  theta1 = reshape(nn_params(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

  theta2 = reshape(nn_params((1 + (hidden_layer * (input_layer + 1))):end), ...
                 output_layer, (hidden_layer + 1));
  
  m = rows(X_train); # Number of data
  
  fprintf('Feedforward Propagation ... ');
  sum_cost = 0;
  acc = 0;
  for i = 1:m # For each input entry
    a1 = X_train(i,:)'; # a1 is a column vector 1024 x 1 with the input values
    a1 = [ones(1,1) ; a1]; # Adding the bias unit - a1 is 1025 x 1
    z2 = theta1 * a1 ./ 255; # Computing z2 - 2048 x 1
    a2 = sigmoid(z2); # Computing the activation vector for layer 2
    a2 = [ones(1,1) ; a2]; # Adding the bias unit, a1 is 2049 x 1
    z3 = theta2 * a2 ./ 255; # Computing z3 - 26 x 1
    a3 = sigmoid(z3); # Computing the activation vector for layer 3
    hyp = a3; # The hypothesis
    [max_el max_index] = max(hyp);
    
    # Compute current y column solution for current input
    y_column = [ zeros(y_train(i,1)-1,1); ones(1,1); zeros(output_layer - y_train(i,1),1) ];
    if (y_column(max_index,1) == 1)
      acc+=1;
    endif
    sum_class = 0;
    for k = 1:output_layer
      sum_class += (y_column(k,1) * log(hyp(k,1)) + (1-y_column(k,1)) * log(1 - hyp(k,1)));
    endfor
    sum_cost += sum_class;
  endfor
  J = (-1) * sum_cost / m;
  
  fprintf('Accuracy %d%%\n', acc / m);

  # Regularization
  reg = lambda / (2*m);

  sum_reg = 0;
  for j = 1:rows(theta1)
    for k = 2:columns(theta1)
      sum_reg += power(theta1(j,k),2);
    endfor
  endfor

  for j = 1:rows(theta2)
    for k = 2:columns(theta2)
      sum_reg += power(theta2(j,k),2);
    endfor
  endfor

  reg *= sum_reg;
  J += reg;

  fprintf('done\n');
  
  fprintf('Backward Propagation ... ');
  
  d1 = 0;
  d2 = 0;

  delta = zeros(m, 2);

  for t = 1:m
    a1 = X_train(i,:)'; # a1 is a column vector 1024 x 1 with the input values
    a1 = [ones(1,1) ; a1]; # Adding the bias unit - a1 is 1025 x 1
    z2 = theta1 * a1 ./ 255; # Computing z2 - 2048 x 1
    a2 = sigmoid(z2); # Computing the activation vector for layer 2
    a2 = [ones(1,1) ; a2]; # Adding the bias unit, a1 is 2049 x 1
    z3 = theta2 * a2 ./ 255; # Computing z3 - 26 x 1
    a3 = sigmoid(z3); # Computing the activation vector for layer 3
    hyp = a3; # The hypothesis
    
    # Compute current y column solution for current input
    y_column = [ zeros(y_train(i,1)-1,1); ones(1,1); zeros(output_layer - y_train(i,1),1) ];
      
    delta3 = a3 - y_column;
    delta2 = (theta2'*delta3) .* (a2 .* (1 - a2));
    delta2 = delta2(2:end);
    delta1 = (theta1'*delta2) .* (a1 .* (1 - a1));
    
    d1 += delta2 * a1';
    d2 += delta3 * a2';
    
  endfor

  theta1_grad = d1 / m;
  theta2_grad = d2 / m;
  
  grad = [theta1_grad(:) ; theta2_grad(:)];
  
  fprintf('done\n');
endfunction