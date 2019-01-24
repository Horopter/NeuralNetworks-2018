import math,copy

def initArrZero(num):
	l = []
	for i in range(num):
		l.append(0)
	return l
	
def initMatrix(rows,cols):
	m = []
	for i in range(rows):
		m.append(initArrZero(cols))
	return m	
	
def sigmoid(x):
	return 1.0/(1.0 + math.e**(-x))
	
def sigmoid_m(x):
	if isinstance(x,list):
		lst = []
		for i in x:
			lst.append(sigmoid_m(i))
		return lst
	else:
		return sigmoid(x)
	
def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))
	
def sigmoid_prime_m(x):
	if isinstance(x,list):
		lst = []
		for i in x:
			lst.append(sigmoid_prime_m(i))
		return lst
	else:
		return sigmoid_prime(x)
		
def transpose(m):
	return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
	
def matmul(A,B):
	result = initMatrix(len(A),len(B[0]))
	for i in range(len(A)):  
		for j in range(len(B[0])): 
			for k in range(len(B)): 
				result[i][j] += A[i][k] * B[k][j]
	return result
	
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
	
def scalarmul(A,B):
	if isinstance(A,list) and (isinstance(B,float) or isinstance(B,int)):
		return scalarmul(B,A)
	if isinstance(B,list):
		lst = []
		for i in B:
			lst.append(scalarmul(A,i))
		return lst
	return A*B
	
def subtract(A,B):
	if isinstance(A,list) and isinstance(B,list) and len(A)==len(B):
		lst = []
		for i in range(len(A)):
			lst.append(subtract(A[i],B[i]))
		return lst
	else:
		return A-B
		
class NN:
	def __init__(self,arr):
		assert len(arr)>1
		l = len(arr)
		input = initArrZero(arr[0]+1)
		input[-1] = 1
		self.layers = []
		self.layers.append(input)
		for i in range(1,l-1):
			lst = initArrZero(arr[i])
			self.layers.append(lst)
		self.layers.append(initArrZero(arr[-1]))
		
		self.weights = []
		for i in range(l-1):
			w = initMatrix(len(self.layers[i]),len(self.layers[i+1]))
			self.weights.append(w)
			
	def feedforward(self):
		for i in range(0,len(self.layers)-1):
		    self.layers[i+1] = sigmoid_m(matmul(self.weights[i],self.layers[i]))
		
	def backprop(self,actual,alpha):
	    self.wd = initArrZero(len(self.weights))
	    self.wdm = initArrZero(len(self.weights))
	    
	    for i in range(len(self.weights)-1,-1,-1):
	        if i == len(self.weights)-1:
	            self.wd[i] = hadamard(subtract(self.layers[-1],actual),sigmoid_prime_m(matmul(self.weights[-1],self.layers[-2])))
	        else:
	            self.wd[i] = hadamard(matmul(transpose(self.weights[i+1]),self.wd[i+1]),sigmoid_prime_m(matmul(self.weights[i],self.layers[i])))
	            
	    for i in range(len(self.weights)-1,-1,-1):
	        t = transpose(self.layers[i])
	        self.wdm[i] = matmul(self.wd[i],t)
	        
	    for i in range(len(self.weights)-1,-1,-1):
	        self.weights[i] = matadd(self.weights[i],hadamard(scalarmul(-1*alpha,self.weights[i]),self.wdm[i]))
	    
	def show(self):
		print "Layers : "
		for p in self.layers:
			print p
		
		print "\n\n\n"
		
		print "Weights : "
		for i in range(len(self.weights)):
			print self.weights[i]
			
		print "\n\n\n"
		
n = NN([2,2,2])
n.layers[0] = [[0.05],[0.1],[1]]
n.weights[0] = transpose([[0.15,0.25],[0.2,0.3],[0.35,0.6]])
n.weights[1] = transpose([[0.4,0.5],[0.45,0.55]])
n.show()
for i in range(1000):
    n.feedforward()
    n.backprop([[0.01],[0.99]],5)
n.show()
