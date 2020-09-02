<h1>Supervised Learning Practice</h1>
<h2>Description</h2>
<p>
  This project was built to test my understandings of neural networks for multiple-class classicifacion problems. Given an image with a letter it can predict what letter of the alphabet it is. The algorithm was trained on approximately 2000 training data for each letter (so 2000x26 = 52000) and managed to reach an accuracy of x% in its training guesses. The data was taken from <a href="https://www.kaggle.com/vaibhao/handwritten-characters">here</a>. 
  </br>
  In order to find the best parameters (Theta) this time I used a function provided by Matlab called <i>fmingc</i> which minimizes a continuous differentiable multivariate function. It requires a cost function that returns the cost of the neural network for a specific set of weights and the gradient of the weights.
</p>
<h2>Results</h2>
<p>
  On the training data the algorithm has an accuracy of 100%. 
  The cross-validation set of data contains 100 * 26 = 2600 entries. It managed to predict correctly around 69% of the data.
</p>
<h2>Concepts practiced</h2>
<ul>
  <li>Multi-Class Classification Problem</li>
  <li>Neural Network</li>
  <li>Feature Scaling using Mean Normalization</li>
  <li>Batch Gradient Descent</li>
</ul>

<h2>How to test your own data?</h2>
In order to test your own data you must do the following:
<ol>
  
  <li>Create a directory called "data" in the root directory</li>
  <li>First make sure you have the two folders provided by the Kaggle dataset (link in the description above). You need these data entries for your algorithm to 
  learn and validate</li>
  <li>Inside of directory "data" create another directory called "Test"</li>
  <li>Inside of directory "Test" you will store all your images you want to test</li>
  <li>Make sure the images that you add are of size 32x32 pixels, format ".png" and have a name of the following form: "test_X_N" where 
  X denotes the letter you drew and N denotes the index of the image (starting from 1)</li>
  <li>Now you must go in "main.m" before the last statement and modify variable "m_test" to be equal to the number of images you want to test</li>
  <li>In the function call "testData" make sure you add as the last parameter the letter you want to find in a string/character format ('A')</li>
  <li>Now run the "main.m" script and wait for the "testData.m" script to run. For each letter you will be prompted with the results and a label, 
  either "SUCCESS" or "FAILED" and some additional info. You must press any key on your keyboard to continue and go to the next letter.</li
</ol>