function [J, grad] = computeCost (nn_params, input_layer, hidden_layer, output_layer, X_train, y_train, lambda)
  # Reshaping parameters
  theta1 = reshape(nn_params(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));
  # 1024 x 1025

  theta2 = reshape(nn_params((1 + (hidden_layer * (input_layer + 1))):end), ...
                 output_layer, (hidden_layer + 1));
                 
  # 26 x 1025
  
  m = rows(X_train); # Number of data
  
  #fprintf('Feedforward Propagation ... ');
  sum_cost = 0;
  
  # VECTORIZATION
  # m = 286
  a1 = [ones(1,columns(X_train')) ;  X_train']; # 1025 x 286
  z2 = theta1 * a1; # 1024 x 286
  a2 = sigmoid(z2);
  a2 = [ones(1, columns(a2)) ; a2]; # 1025 x 286
  z3 = theta2 * a2; # 26 x 286
  a3 = sigmoid(z3);
  hyp = a3;  # 26 x 286 - so each column is an output vector
  
  # y_train is a 286 x 1 vector
  y_train = y_train'; # y_train is a 1 x 268 vector
  # for each column we want to add a column vector with the output representation
  y_columns = zeros(26, m);
  for i = 1:columns(y_train)
    
    value = y_train(1,i); # Take the resulting value for output y_i
    # Create a column vector representing the value
    y_column = [ zeros(value,1) ; ones(1,1) ; zeros(26 - value - 1,1)];
    # Add the column to the y_columns matrix
    y_columns(:,i) = y_column;
    
  endfor
  
  sum_class = 0;
  prod = y_columns * X_train;  
  sum_cost = sum(sum(prod));
  J = sum_cost / m;
  
  # Regularization
  reg1 = sum(sum(theta1(1:end,2:end).^2));
  reg2 = sum(sum(theta2(1:end,2:end).^2));
  
  reg = lambda / (2*m);

  reg *= (reg1 + reg2);
  J += reg;

  #fprintf('done\n');
  
  #fprintf('Backward Propagation ... ');
  
  d1 = 0;
  d2 = 0;
  
  delta = zeros(m, 2);
  
  # VECTORIZATION
  # m = 286
  a1 = [ones(1,columns(X_train')) ;  X_train']; # 1025 x 286
  z2 = theta1 * a1; # 1024 x 286
  a2 = sigmoid(z2);
  a2 = [ones(1, columns(a2)) ; a2]; # 1025 x 286
  z3 = theta2 * a2; # 26 x 286
  a3 = sigmoid(z3);
  hyp = a3;  # 26 x 286 - so each column is an output vector

  # y_train is already a 1 x 268 vector
  # for each column we want to add a column vector with the output representation
  y_columns = zeros(26, m);
  for i = 1:columns(y_train)
    
    value = y_train(1,i); # Take the resulting value for output y_i
    # Create a column vector representing the value
    y_column = [ zeros(value,1) ; ones(1,1) ; zeros(26 - value - 1,1)];
    # Add the column to the y_columns matrix
    y_columns(:,i) = y_column;
    
  endfor
  
  delta3 = a3 - y_columns;
  delta2 = theta2'*delta3 .* (a2 .* (1-a2));
  delta2 = delta2(2:end, :);
  delta1 = theta1'*delta2 .* (a1 .* (1-a1));
  
  d1 += delta2 * a1';
  d2 += delta3 * a2';

  theta1_grad = d1 / m;
  theta2_grad = d2 / m;
  
  theta1_grad(:, 2:end) += lambda / m * theta1(:, 2:end);
  theta2_grad(:, 2:end) += lambda / m * theta2(:, 2:end);
  
  grad = [theta1_grad(:) ; theta2_grad(:)]; # MODIFIES SUCCESFULLY, BUT SLOW AF (0.xxx DIFFERENCE)
  
  #fprintf('done\n');
endfunction