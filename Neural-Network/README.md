An example neural network learning

data: ex4data1.mat is a hand-written digits training set contains 5000 training examples in native Octave matrix format
  - each training example is a 20 by 20 grayscale image, unrolled into a 400-dimensional vector
  - label y for each training example
initial weights: ex3weights.mat contains pre-defined set of network parameters
  - layer1: 25 x 401
  - layer2: 10 x 26

goal: implement neural networks to recognize hand-written digits

#### other files:
- displayData.m :Function to help visualize the dataset
- fmincg.m :Function minimization routine
- computeNumericalGradient.m :Numerically compute gradients
- checkNNGradients.m :Function to help check your gradients
- predict.m :Neural network prediction function
