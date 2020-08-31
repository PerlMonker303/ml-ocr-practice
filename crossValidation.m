function acc = crossValidation (X_cv, y_cv, theta1, theta2, output_layer)
  
  m_cv = rows(X_cv);
  acc = 0;
  
  for i = 1:m_cv # For each input entry
    a1 = X_cv(i,:)'; # a1 is a column vector 1024 x 1 with the input values
    tst1 = reshape(a1, 32, 32)
    a1 = [ones(1,1) ; a1]; # Adding the bias unit - a1 is 1025 x 1
    z2 = theta1 * a1; # Computing z2 - 1024 x 1
    m2 = max(z2);
    z2 = z2 ./ m2;
    a2 = sigmoid(z2); # Computing the activation vector for layer 2
    a2 = [ones(1,1) ; a2]; # Adding the bias unit, a1 is 1025 x 1
    z3 = theta2 * a2; # Computing z3 - 26 x 1
    m3 = max(z3);
    z3 = z3 ./ m3;
    tst3 = reshape(z3, 2, 13) # contains a strange 1 value at pos 14
    a3 = sigmoid(z3); # Computing the activation vector for layer 3
    hyp = z3; # The hypothesis
    
    [max_el max_index] = max(hyp);
    
    # Compute current y column solution for current input
    y_column = [ zeros(y_cv(i,1)-1,1); ones(1,1); zeros(output_layer - y_cv(i,1),1) ];
    
    pause
    if (y_column(max_index) == 1)
      acc+=1
    endif
  endfor
  
  acc = acc / m_cv;
  
endfunction
