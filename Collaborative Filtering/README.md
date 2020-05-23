An example Collaborative Filtering Model
 
data: 
- ex8_movies.mat: matrix type data in native Octave format
  - Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on  943 users
  - R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i, 0 otherwise
- ex8_movieParams.mat: data with pre-trained weights in native octave format
  - contains: X, Theta, num_users, num_movies, num_features
- movie_idx.txt: list of all movies and their number

goals: Recommender Systems
- implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings

other files:
- checkCostFunction.m :Gradient checking for collaborative filtering
- computeNumericalGradient.m :Numerically compute gradients
- fmincg.m :Function minimization routine (similar to fminunc)
