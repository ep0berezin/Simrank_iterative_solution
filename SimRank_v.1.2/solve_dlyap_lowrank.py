import numpy as np
from scipy.sparse import csr_matrix
import sys
import time
import matplotlib.pyplot as plt 
import datetime as dt
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile

class G_hat_operator:
	def __init__(self, A, c):
		self.A = A
		self.n = self.A.shape[1]
		if (self.n!=self.A.shape[0]):
			print(f"Warning! Non-zero adjacency matrix detected when constructing operator.")
		self.c = c
	def __call__(self, M_k, Q, rank):
		U, s, V = np.linalg.svd(M_k, full_matrices = False)
		U_r = U[:,:rank]
		S_r = np.diag(s[:rank])
		V_r = V[:rank]
		print("Truncated with rank ", rank, " U_r dim:", U_r.shape, " V_r dim ", V_r.shape)
		T_hat = self.A.T@U_r@S_r@V_r@self.A
		np.fill_diagonal(T_hat, 0) #T- diag(T)
		M_kp1 = Q + self.c*T_hat
		return M_kp1
		
def RecTrunc(acc, k_iter_max, A, c):
	n = A.shape[0]
	Q = A.T@A
	Q = c*( Q - np.diag(np.diag(Q)) ) #cA.TA-c*diag(A.TA)
	Q_csr = csr_matrix(Q)
	I = np.identity(n)
	M_k = I
	solutions = []
	G_hat = G_hat_operator(csr_matrix(A), c)
	for rank in [n, n//10, n//20]:
		r = rank
		M_k = I
		print("Truncating with rank", r)
		for k in range(k_iter_max):
			M_km1 = M_k
			M_k = G_hat(M_k, Q_csr, r)
			relres = np.linalg.norm(M_k-M_km1)/np.linalg.norm(M_km1)
			print("Relative residual: ", relres)
			if (relres < acc):
				solutions.append(M_k)
				break
	
	print("Solutions number: ", len(solutions))
	for k in range(1,len(solutions)):
		error_matrix = abs((solutions[0]-solutions[k]))
		print("Error:", np.linalg.norm(error_matrix.reshape(n**2, order = 'F'), ord = 1 )/n**2)
		print("Maximum error: ", np.max(error_matrix))
	plt.figure()
	graph_log = plt.imshow(np.log(M_k+1e-15))
	cbar = plt.colorbar()
	cbar.set_label("ln(S[i,j])")
	plt.title(f"Матрица S, логарифмическая шкала", fontweight = "bold")
	plt.figure()
	graph = plt.imshow(M_k)
	cbar = plt.colorbar()
	cbar.set_label("(S[i,j])")
	plt.title(f"Матрица S", fontweight = "bold")
	plt.show()
