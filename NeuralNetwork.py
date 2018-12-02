import math,copy

def initArrZero(num):
	l = []
	for i in range(num):
		l.append(0)
	return l
	
def initMatrix(rows,cols,increment):
	m = []
	for i in range(rows):
		m.append(initArrZero(cols))
	if increment:
		for i in range(rows):
			m[i].append(0)
	return m	
	
def sigmoid(x):
	return 1.0/(1.0 + math.e**(-x))
	
def sigmoid_m(x):
	if isinstance(x,list):
		for i in x:
			return sigmoid_m(x)
	else:
		return sigmoid(x)
	
def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))
	
def sigmoid_prime_m(x):
	if isinstance(x,list):
		for i in x:
			return sigmoid_prime_m(x)
	else:
		return sigmoid_prime(x)
		
def transpose(m):
	return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
	
def matmul(A,B):
	return [[sum(a * b for a, b in zip(A_row, B_col))  
			B_col in zip(*B)] 
				for A_row in A]
	
def matadd(X,Y):
	result = copy.deepcopy(X)
	for i in range(len(X)): 
		for j in range(len(X[0])): 
			result[i][j] = X[i][j] + Y[i][j]
	return result
	
def hadamard(X,Y):
	result = copy.deepcopy(X)
	for i in range(len(X)): 
		for j in range(len(X[0])): 
			result[i][j] = X[i][j] * Y[i][j]
	return result
	
	
class NN:
	def __init__(self,arr):
		assert len(arr)>1
		l = len(arr)
		input = initArrZero(arr[0]+1)
		input[-1] = 1
		self.layers = []
		self.layers.append(input)
		for i in range(1,l-1):
			lst = initArrZero(arr[i]+1)
			lst[-1] = 1
			self.layers.append(lst)
		self.layers.append(initArrZero(arr[-1]))
		
		self.weights = []
		for i in range(l-1):
			w = initMatrix(arr[i]+1,arr[i+1],i!=l-2)
			self.weights.append(w)

		for i in range(len(self.layers)):
			print self.layers[i]
		
		for i in range(len(self.weights)):
			print self.weights[i]
			
	def feedforward(self):
		for i in range(1,len(self.layers)):
			self.layers[i] = sigmoid_m(matmul(transpose(self.weights[i]),self.layers[i-1]))
		
	def backprop(self,actual,alpha):
		self.wd = []
		last = hadamard((self.layers[-1] - actual),sigmoid_prime_m(self.layers[-1]))
		for i in range(len(self.weights)-1,-1,-1):
			if i == len(self.weights)-1:
				self.wd[i] = last
			else:
				self.wd[i] = matmul(transpose(self.weights[i+1]),self.wd[i-1])
		for i in range(len(self.weights)-1,-1,-1):
			self.wd[i] = matmul(self.wd[i],transpose(self.layers[i-1]))
			
		self.weights = matadd(self.weights,self.wd)
		
NN([3,4,2])
		