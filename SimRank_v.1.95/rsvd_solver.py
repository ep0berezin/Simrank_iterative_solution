import numpy as np
import time

class iterations_data:
	def __init__(self):
		self.iterations = []
		self.residuals = []
		self.k_iter = 0
	def __call__(self, r, printout = True):
		if printout: print(f"Iteration: {self.k_iter}")
		self.k_iter+=1
		self.iterations.append(self.k_iter)
		self.residuals.append(r)
		if printout: print('Current relative residual =', r)
		return r

class F_M_operator: #for RSVD iterations.
	def __init__(self, A, c):
		self.A = A
		self.n = self.A.shape[1]
		if (self.n!=self.A.shape[0]):
			print(f"Warning! Non-square adjacency matrix detected when constructing operator.")
		self.c = c
		B = c*(A.T@A) #I to auto-convert to dense matrix to use fill then.
		np.fill_diagonal(B, 0.0)
		self.B = B
	def __call__(self,M):
		T_1 = self.c*(self.A.T@M@self.A)
		np.fill_diagonal(T_1, 0.0)
		return T_1+self.B
	def randomize(self, U_hat, r, p): #U_hat = eigvecs matrix U multiplied by sqrt(Sigma), r = target rank, p = oversampling parameter
		Omega = np.random.standard_normal((self.n, r+p))
		MOmega = np.empty((self.n, r+p))
		d = np.empty(self.n) #reserving diagonal vector
		T_1 = U_hat.T@self.A #U_hat.T@A
		T_2 = T_1.T
		for i in range(r+p):
			w = Omega[:,i]
			t_1 = T_1@w #chain of matvecs
			t_2 = (T_2@t_1)*self.c
			for q in range(self.n): #optimal diag using symmetric nature of (U.T@A).T @ (U.T@A)
				d[q] = np.dot(T_1[:,q],T_1[:,q])*w[q]
			MOmega[:,i] = t_2 - self.c*d + self.B@w
		return MOmega

def RSVDIters(A, c, r, p, k_max_iter, eps): #RSVD iterations based on Oseledets article
	np.random.seed(42)
	n = A.shape[0]
	iterdata = iterations_data()
	F_M = F_M_operator(A,c) #init operator
	st = time.time()
	M_0 = np.zeros((n,n))
	M_prev = F_M(M_0) #obtain M_1 = B
	U, sigma, V = np.linalg.svd(M_prev, full_matrices = False)
	U_hat = U*np.sqrt(sigma) #Get U_hat = U@sqrt(Sigma); store this way to effectively compute diag()*w_i.
	print(f"CHECK U_0: {np.linalg.norm(U_hat)}")
	for k in range(k_max_iter):
		MOmega = F_M.randomize(U_hat, r, p) #effective multiplication by Omega
		print(f"CHECK MOmega: {np.linalg.norm(MOmega)}")
		Q, R = np.linalg.qr(MOmega) #QR decompo
		t_QTM = Q.T@M_prev #obtain Q^T @ M
		print(f"CHECK QTMprev: {np.linalg.norm(t_QTM)}")
		M_r = Q@t_QTM #M_r
		U_0, sigma, V = np.linalg.svd(t_QTM, full_matrices = False) #small SVD
		U = (Q@U_0)[:,:r] #get U and truncate to r
		U_hat = U*np.sqrt(sigma[:r]) #saving U_hat
		print(f"CHECK U result: {np.linalg.norm(U_hat)}")
		relres = np.linalg.norm((M_r-M_prev), ord = 'fro')/np.linalg.norm((M_prev), ord = 'fro')
		iterdata(relres)
		if (relres < eps):
			break
		M_prev = M_r
	et = time.time()
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return M_prev, solutiondata
