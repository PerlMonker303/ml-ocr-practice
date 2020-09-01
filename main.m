# Clear and Close Figures
clear ; close all; clc
pkg load image;

fprintf('Loading data ... ');

### Load Training Data ###

max_training_data = 10; # 2000
m = max_training_data * 26;
X_train = zeros(m, 32*32); # Create a matrix in which you will store the data
y_train = zeros(m, 1);
k = 1;
for i = 0 : 25
  letter = char(i+65);
  #fprintf("Letter %c ...\n", letter);
  f = 1;
  for j = 1:max_training_data
    try
      string = strcat('./data/Train/', letter ,'/', num2str(j), '.jpg');
      [url_img] = imread(string); # Images are of size 32 x 32
      # Vectorize the image
      vec_img = reshape(url_img, 1, 32*32); # As a row vector
      X_train(k,:) = vec_img;
      y_train(k,:) = i;
      k+=1;
    catch
      f+=1; # Marking fails
    end_try_catch
  endfor
  
  # f = number of missing images
  j = max_training_data + 1;
  #fprintf('%d failures for letter %c  ', f, i+65);
  while (f > 0)
    try
      string = strcat('./data/Train/', letter ,'/', num2str(j), '.jpg');
      [url_img] = imread(string); # Images are of size 32 x 32
      # Vectorize the image
      vec_img = reshape(url_img, 1, 32*32); # As a row vector
      X_train(k,:) = vec_img;
      y_train(k,:) = i;
      k+=1;
      f-=1; # Decrease the number of failures
    catch
      
    end_try_catch
    j+=1;
  endwhile
  #fprintf('... done\n');
endfor
fprintf('done\n');

# Feature scaling
fprintf('Scaling features ... ');
[X_train mu sigma] = featureScaling(X_train);
fprintf('done\n');

# Design the Neural Network
input_layer = 32*32; # 1024
hidden_layer = input_layer; # 1024
output_layer = 26; # vector of 1s and 0s with 26 rows

theta1 = zeros(hidden_layer, input_layer + 1);
theta2 = zeros(output_layer, hidden_layer + 1);

# Randomly assign values to parameters theta1, theta2
[theta1 theta2] = assignParameters(theta1, theta2);

# Unroll parameters 
initial_nn_params = [theta1(:) ; theta2(:)];

# Training the Neural Network
number_iterations = 20;
options = optimset('MaxIter', number_iterations);

lambda = 0.03;

# Create "short hand" for the cost function to be minimized
costFunction = @(p) computeCost(p, input_layer, hidden_layer, output_layer, X_train, y_train, lambda);

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
fprintf('STARTED OPTIMIZATION\n');
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

fprintf('FINISHED OPTIMIZATION\n');
# Reshape the optimal parameters
theta1 = reshape(nn_params(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

theta2 = reshape(nn_params((1 + (hidden_layer * (input_layer + 1))):end), ...
                 output_layer, (hidden_layer + 1));

                 
# Now we cross-validate the obtained optimal thetas and compute the accuracy

### Load Validation Data ###
fprintf("Started Cross-Validation ...\n");

max_cv_data = 10; # 100
m_cv = max_cv_data * 26;
X_cv = zeros(m_cv, 32*32); # Create a matrix in which you will store the data
y_cv = zeros(m_cv, 1);
k = 1;
for i = 0 : 25
  letter = char(i+65);
  #fprintf("Letter %c ...\n", letter);
  f = 1;
  for j = 1:max_cv_data
    try
      string = strcat('./data/Validation/', letter ,'/', num2str(j), '.jpg');
      [url_img] = imread(string); # Images are of size 32 x 32
      # Vectorize the image
      vec_img = reshape(url_img, 1, 32*32); # As a row vector
      X_cv(k,:) = vec_img;
      y_cv(k,:) = i;
      k+=1;
    catch
      f+=1; # Marking fails
    end_try_catch
  endfor
  
  # f = number of missing images
  j = max_cv_data + 1;
  #fprintf('%d failures for letter %c  ', f, i+65);
  while (f > 0)
    try
      string = strcat('./data/Validation/', letter ,'/', num2str(j), '.jpg');
      [url_img] = imread(string); # Images are of size 32 x 32
      # Vectorize the image
      vec_img = reshape(url_img, 1, 32*32); # As a row vector
      X_cv(k,:) = vec_img;
      y_cv(k,:) = i;
      k+=1;
      f-=1; # Decrease the number of failures
    catch
      
    end_try_catch
    j+=1;
  endwhile
  #fprintf('... done\n');
endfor
fprintf('... done\n');

# Feature scaling
fprintf('Scaling features ... ');
for row = 1:rows(X_cv)
  for col = 1:columns(X_cv)
    X_norm(row, col) = (X_cv(row, col) - mu(1,col)) / sigma(1,col);
  endfor
endfor
fprintf('done\n');

# computeCostValidation() => feedforward and check if the results match the y column
accuracy = crossValidation(X_cv, y_cv, theta1, theta2, output_layer);
fprintf('Accuracy on cv: %d%%\n', accuracy);