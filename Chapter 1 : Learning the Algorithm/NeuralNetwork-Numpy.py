import numpy as np

def initEmpty(num):
	l = []
	for i in range(num):
		l.append([])
	return l

def initArrZero(num):
	return np.zeros(num,dtype=int)
	
def initMatrix(rows,cols):
	return np.zeros([rows,cols],dtype=int)	
	
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
	
def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))
		
def transpose(m):
	return np.transpose(m)
	
def matmul(A,B):
	return np.matmul(A,B)

def matadd(X,Y):
	return np.add(X,Y)
	
def hadamard(X,Y):
	return np.multiply(X,Y)

def scalarmul(A,B):
	return A*B
	
def subtract(A,B):
	return np.subtract(A,B)

def mkArr(A):
	return np.asarray(A)
		
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
		    self.layers[i+1] = sigmoid(matmul(self.weights[i],self.layers[i]))
		
	def backprop(self,actual,alpha):
	    self.wd = initEmpty(len(self.weights))
	    self.wdm = initEmpty(len(self.weights))
	    for i in range(len(self.weights)-1,-1,-1):
	        if i == len(self.weights)-1:
	            self.wd[i] = hadamard(subtract(self.layers[-1],actual),sigmoid_prime(matmul(self.weights[-1],self.layers[-2])))
	        else:
	            self.wd[i] = hadamard(matmul(transpose(self.weights[i+1]),self.wd[i+1]),sigmoid_prime(matmul(self.weights[i],self.layers[i])))
	            
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
n.layers[0] = mkArr([[0.05],[0.1],[1]])
n.weights[0] = transpose(mkArr([[0.15,0.25],[0.2,0.3],[0.35,0.6]]))
n.weights[1] = transpose(mkArr([[0.4,0.5],[0.45,0.55]]))
n.show()
for i in range(10):
    n.feedforward()
    n.backprop(mkArr([[0.01],[0.99]]),5)
n.show()