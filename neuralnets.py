import numpy as np
from numpy import random
import math

def sigmoid(x):
	return 1/(1+math.exp(-x))

def diffsig(x):
	return sigmoid(x)*(1 - sigmoid(x))


f = open('digits.wdep')
fp = f.read().splitlines()
n = 1
wij = random.rand(65, 10)
wjk = random.rand(10, 2)
print wij
print wjk
while (32*n+n-1<=int(len(fp))):
	for i in fp[32*n+n-1]:
		if ( i == '5' or i == '0' or i=='2' ):
			#print i
			if i == '0':
				Tk = np.array([0, 0])
			if i == '2':
				Tk = np.array([0, 1])
			if i == '5':
				Tk = np.array([1, 1])
			array = [[0 for i in range(32)] for j in range(32)]
			a = 0
			for j in range(32*n-32+n-1,32*n+n-1):
				b = 0
				for k in fp[j]:
					array[a][b] = float(k)

					b = b +1
				a= a+1
			pre_array = [0 for i in range(65)]
			k = 1
			for i in range(0,32,4):
				for j in range(0,32,4):
					sum = 0
					for a in range(4):
						for b in range(4):
							sum += array[i+a][j+b]

					pre_array[k] = sum/16
					k = k+1 

			Xi = np.array(pre_array)
			Xi = Xi[np.newaxis]
			Tk = Tk[np.newaxis]

			iti = 1000
			while(iti):
				netj = np.dot(Xi, wij)
				Yj = random.rand(1,10)
				fdnetj = random.rand(1,10)
				for i in range(len(netj[0])):
					Yj[0][i] = sigmoid(netj[0][i])
					fdnetj[0][i] = diffsig(netj[0][i])
				netk = np.dot(Yj,wjk)

				Zk = random.rand(1,2)
				fdnetk = random.rand(1,2)
				for i in range(len(netk[0])):
					Zk[0][i] = sigmoid(netk[0][i])
					fdnetk[0][i] = diffsig(netk[0][i])

				deltak = np.multiply((Tk-Zk), fdnetk)
				deltaj = np.multiply(np.sum(np.dot(wjk,deltak.T)),fdnetj) 
				Dwjk = np.dot(Yj.T, deltak)
				Dwij = np.dot(Xi.T, deltaj)

				eta = 0.02

				wjk = wjk + np.multiply(eta,Dwjk)
				wij = wij + np.multiply(eta,Dwij)
				iti = iti - 1
			#print Zk
	n = n+1

#print wjk
#print Tk-Zk





	


