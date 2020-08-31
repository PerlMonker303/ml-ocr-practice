function [theta1, theta2] = assignParameters(theta1, theta2)
  INT_EPSILON = 3;
  theta1 = rand(size(theta1)) * (2*INT_EPSILON) - INT_EPSILON;
  theta2 = rand(size(theta2)) * (2*INT_EPSILON) - INT_EPSILON;
endfunction
