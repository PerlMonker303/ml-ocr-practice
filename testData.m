function retval = testData (m_test, theta1, theta2, output_layer, letter)
  # We will be given the letter as a parameter
  # First we read the data
  
  fprintf("Started Testing Data ...");
  
  X_cv = zeros(m_test, 32*32); # Create a matrix in which you will store the data
  y_cv = zeros(m_test, 1);
  k = 1;
  mx = 0;
  # We read m_test letters of type 'letter'
  for i = 1:m_test
    string = strcat('./data/Test/test_', letter ,'_', num2str(i), '.png');
    [url_img] = imread(string); # Images are of size 32 x 32
    url_img = double(url_img);
    color_r = url_img(:,:,1);
    color_g = url_img(:,:,2);
    color_b = url_img(:,:,3);
    avg = (color_r + color_g + color_b) / 3; # merge
    
    # Vectorize the image
    vec_img = reshape(avg, 1, 32*32); # As a row vector
    X_test(i,:) = vec_img;
    y_test(i,:) = letter - 65;
  endfor
  
  
  # VECTORIZATION
  a1 = [ones(1,columns(X_test')) ;  X_test']; # 1025 x m_test
  z2 = theta1 * a1; # 1024 x m_test
  a2 = sigmoid(z2);
  a2 = [ones(1, columns(a2)) ; a2]; # 1025 x m_test
  z3 = theta2 * a2; # 26 x m_test
  a3 = sigmoid(z3);
  hyp = a3;  # 26 x m_test - so each column is an output vector
  
  [max_value max_index] = max(hyp); # max_index contains the results of your prediction
  # size 1 x 1 (=m_test)
  # Now we compare these results to the output_layer values
  
  for i = 1:m_test
    value = max_index(1,i);
    fprintf('For image with index %d predicted letter %c, actual letter: %c ', i, char(value-1+65), char(y_cv(i,1) + 65));
    if (value - 1 == y_cv(i,1))
      fprintf('SUCCESS\n');
    else
      fprintf('FAIL\n');
    endif
    fprintf('Press any key to continue ...\n');
    pause
  endfor
  
  fprintf('Finished Testing Data\n');
endfunction
