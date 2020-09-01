function [X_norm, mu, sigma] = featureScaling(X)
  
# Initialise mu and sigma
X_norm = X;
mu = zeros(1, columns(X));
sigma = zeros(1, columns(X));


mu = mean(X_norm);
sigma = std(X_norm);

# Go through the entries and subtract the mean value, then divide by the std deviation
#for col = 1:columns(X_norm)
  #X_norm(:, col) = (X_norm(:, col) - mu(1,col)) ./ sigma(1,col);
#endfor


#X_norm - holds the result
X = X_norm ./ 255;

# We store sigma and miu such that when we want to predict the price for a new house
# we first have to normalize it using these values

endfunction