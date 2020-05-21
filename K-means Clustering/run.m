%% An example K-Means Clustering
%
% data:
% - ex7data2.mat: 2D matrix dataset representing coordinates of points
% - bird_small.png: a 128 x 128 sample image for compression
%
% goal: implement Kmeans clustering to compress a colored image to 16 color groups
%

%% Initialization
clear ; close all; clc

%% ================= Find Closest Centroids ====================
%  We have divided the learning algorithm 
%  into two functions -- findClosestCentroids and computeCentroids. 
%

%% ---------------------------------------------------------------- %%
function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% initialize return variable
idx = zeros(size(X,1), 1);

for r=1:length(idx)
	distance = (norm(X(r,:) - centroids(1,:)))^2;
	idx(r) = 1;
	for s=2:K
		if (norm(X(r,:) - centroids(s,:)))^2 < distance
			distance = (norm(X(r,:) - centroids(s,:)))^2;
			idx(r) = s;
		endif
	end;
end;

end
% ---------------------------------------------------------------- %

fprintf('Finding closest centroids.\n\n');

% Load an example dataset that we will be using
load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the
% initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');


%% ===================== Compute Means =========================

%% ---------------------------------------------------------------- %%
function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% initialize return variable
centroids = zeros(K, n);

for c=1:K
	rowx = idx==c;
	sumx = rowx'*X;
	centroids(c,:)=sumx/sum(rowx);
end;

end
% ---------------------------------------------------------------- %

fprintf('\nComputing centroids means.\n\n');

%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');


%% =================== K-Means Clustering ======================
%  Run the K-Means algorithm on
%  the example dataset we have provided. 
%
fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Load an example dataset
load('ex7data2.mat');

% Settings for running K-Means
K = 3;
max_iters = 10;

% For consistency, here we set centroids to specific values
% but in practice you want to generate them automatically, such as by
% settings them to be random examples (as can be seen in
% kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells our function to plot
% the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');


%% ---------------------------------------------------------------- %%
function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

centroids = zeros(K, size(X, 2));

% Initialize the centroids to be random examples

% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));

% Take the first K examples as centroids
centroids = X(randidx(1:K), :);

end
% ---------------------------------------------------------------- %

%% ============= K-Means Clustering on Pixels ===============
%  Use K-Means to compress an image by first run K-Means on 
%  the colors of the pixels in the image and
%  then map each pixel onto its closest centroid.
%  

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
%  This creates a three-dimensional matrix A whose first two indices 
%  identify a pixel position and whose last index represents red, 
%  green, or blue. For example, A(50, 33, 3) gives the blue intensity 
%  of the pixel at row 50 and column 33.
A = double(imread('bird_small.png'));

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run K-Means algorithm on this data
% You should try different values of K and max_iters here
K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids
% randomly. 
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);


%% ================= Image Compression ======================
%  Use the clusters of K-Means to compress an image. To do this, 
%  we first find the closest clusters for each example. 

fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Essentially, now we have represented the image X as in terms of the
% indices in idx. 

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));

