An example Support Vector Machines

data: 
  - ex6data1.mat, ex6data2.mat, ex6data3.mat
    - are sets of coordinates and their label in native Octave matrix format
    - each coordinate contains X1 and X2
    - label y for positive or negative
    - ex6data3.mat also have a subset of data points for validation
  - spamTrain.mat, spamTest.mat
    - training and test spam emails and their label in native Octave format

figures:
  - svm1.png: dataset1 with trained linear SVM model
  - svm2.png: dataset2 with trained SVM model
  - svm3.png: dataset3 with trained SVM model

goal: implement SVM and train models for binary classification
  - including feature extraction from sample emails
