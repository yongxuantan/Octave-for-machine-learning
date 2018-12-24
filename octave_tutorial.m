syntax:
	% : comment
	~= : not equal
	&& : and
	|| : or
	; : suppress output
	disp('one') : print one

definition:
	1 : true
	0 : false
	
Matrix:
	A = [1 2; 3 4] : create 2 by 2 matrix
	A(2,2) : load value of at position of matrix
	A(2,:) : load all columns of that row
	A([1 2],:) : load specified rows and all columns
	A(:,2) = [2;3] : assign new value
	A = [A, [5;6]] : append another column vector to the right
	A(:) : list all values in vector form
	V = 1:0.1:2 : creates vector with values between 1 and 2, at 0.1 increments
	ones(2,3) : create a 2x3 matrix filled with 1s, also works for zeros, rand
	eye(4) : create a 4x4 identity matrix
	help eye : show documentation for eye function
	size(A) : return dimension of A matrix
	length(A) : return longest dimension of A matrix
	
files:
	common data files: .dat, .mat, .txt
	load <filename.ext> : load file
	load('<filename.ext>') : load file
	save <filename.ext> <name> : same content of NAME into FILENAME.EXT
	who/whos : show variables
	clear/clear <name> : remove all/specific variables
	
computation:
	A < 3 : element wise comparision
	A.*B : element wise multiplication
	A' : transpose matrix
	sum(A) : sum all elements
	[val, ind] = max(A) : return max value and index of the value
	magic(3) : try it ;)

iteration:
	if x==1, <actions>; else if x==2, <actions>; else <actions>; end;
	for i=1:10, <actions>; end;
	while i<=5, <actions>; break; end;
	
functions:

plotting:

% to be continued