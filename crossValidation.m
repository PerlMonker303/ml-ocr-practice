function acc = crossValidation (X_cv, y_cv, theta1, theta2, output_layer)
  
  m_cv = rows(X_cv);
  acc = 0;
  
  # VECTORIZATION
  # m = 286
  a1 = [ones(1,columns(X_cv')) ;  X_cv']; # 1025 x 286
  z2 = theta1 * a1; # 1024 x 286
  a2 = sigmoid(z2);
  a2 = [ones(1, columns(a2)) ; a2]; # 1025 x 286
  z3 = theta2 * a2; # 26 x 286
  a3 = sigmoid(z3);
  hyp = a3;  # 26 x 286 - so each column is an output vector
  
  [max_value max_index] = max(hyp); # max_index contains the results of your prediction
  # size 1 x 286 (=m_cv)
  # Now we compare these results to the output_layer values
  
  for i = 1:m_cv
    value = max_index(1,i);
    if (value == y_cv(i,1))
      acc+=1;
    endif
  endfor
  
  acc = acc / m_cv;
  
endfunction
