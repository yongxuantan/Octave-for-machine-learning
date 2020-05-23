An example Anomaly Detection

data: 
- ex8data1.mat
  - X: 307 unlabeled examples, a 2D dataset with features
    - throughput (mb/s), latency (ms)
  - Xval and yval for cross validation
- ex8data2.mat
  - X: an 11-D dataset of server features
  - Xval and yval for cross validation

figures:
- anomaly1.png: The Gaussian distribution contours of the distribution fit to the dataset
- anomaly2.png: The classified anomalies of ex8data1.mat validation set

goals: Anomaly Detection
- implement an anomaly detection algorithm to detect anomalous behavior in server computers by fitting a Gaussian model
  
other files:
- multivariateGaussian.m :Computes the probability density function for a Gaussian distribution
- visualizeFit.m :2D plot of a Gaussian distribution and a dataset
