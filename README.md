<h1>Supervised Learning Practice</h1>
<h2>Description</h2>
<p>
  This project was built to test my understandings of neural networks for multiple-class classicifacion problems. Given an image with a letter it can predict what letter of the alphabet it is. The algorithm was trained on approximately 2000 training data for each letter (so 2000x26 = 52000) and managed to reach an accuracy of x% in its training guesses. The data was taken from <a href="https://www.kaggle.com/vaibhao/handwritten-characters">here</a>. 
  </br>
  In order to find the best parameters (Theta) this time I used a function provided by Matlab called <i>fmingc</i> which minimizes a continuous differentiable multivariate function. It requires a cost function that returns the cost of the neural network for a specific set of weights and the gradient of the weights.
</p>
<h2>Results</h2>
<p>
  The cross-validation set of data contains 100 * 26 = 2600 entries. It managed to predict 0.03% correct so far.
</p>
<h2>Concepts practiced</h2>
<ul>
  <li>Multi-Class Classification Problem</li>
  <li>Neural Network</li>
  <li>Feature Scaling using Mean Normalization</li>
  <li>Batch Gradient Descent</li>
  
</ul>