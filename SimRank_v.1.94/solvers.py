import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import scipy.sparse as scsp
import sys
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
		

#---Fixed Point---

def SimpleIter(LinOp, tau, x_0, b, k_max, printout, eps = 1e-13):
	N = x_0.shape[0] #N = n^2
	s = x_0
	iterdata = iterations_data()
	st = time.time()
	for k in range(k_max):
		s_prev = s
		s = (b-LinOp(s))*tau+s
		r_norm2 = np.linalg.norm(s-s_prev, ord = 2)
		relres = r_norm2/np.linalg.norm(b, ord = 2)
		iterdata(relres, printout)
		if (relres  < eps):
			break
	et = time.time()
	elapsed = et - st
	if printout: print(f"Average iteration time: {elapsed/iterdata.iterations[-1]} s")
	if printout: print(f"Elapsed: {elapsed} s")
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return s, solutiondata
#---


#---GMRES(m)---

def LSq(beta, H):
	m = H.shape[1]
	e1 = np.zeros((m+1,1))
	e1[0,0] = 1.0 #generating e1 vector
	b = e1*beta
	#y = np.linalg.inv(H.T@H)@H.T@b
	y = np.linalg.pinv(H)@b
	return y

def Arnoldi(V_list, h_list, m_start, m_Krylov, LinOp, eps_zero = 1e-15, printout = False):
	for j in range(m_start, m_Krylov):
		#print(f"Building Arnoldi: V[:,{j}]")
		st_1 = time.time()
		v_j = V_list[j]
		w_j = LinOp(v_j).reshape(-1, order='F') #
		if printout: print("Evaluate Av_j time:", time.time()-st_1)
		Av_j_norm2 = np.linalg.norm(w_j, ord = 2)
		st_2 = time.time()
		for i in range(j+1):
			v_i = V_list[i]
			h_ij = v_i@w_j  
			h_list[i][j] = h_ij
			w_j -= h_ij*v_i
		if printout: print("MGS for v_{j+1} time:", time.time()-st_2)
		w_j_norm2 = np.linalg.norm(w_j, ord = 2)
		h_list[j+1][j] = w_j_norm2
		if (w_j_norm2 <= eps_zero):
			return j
		V_list[j+1] = w_j*(1/w_j_norm2)
	return m_Krylov

def GMRES_m(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13, printout = False):
	N = x_0.shape[0] #N = n^2
	r = b - LinOp(x_0)
	r_norm2 = np.linalg.norm(r, ord = 2)
	relres = r_norm2/np.linalg.norm(b, ord = 2)
	iterdata = iterations_data()
	iterdata(relres, printout)
	st = time.time()
	x = x_0
	break_outer = False
	for k in range(k_max):
		if (break_outer):
			break
		st_restart = time.time()
		r = b - LinOp(x)
		r_norm2 = np.linalg.norm(r, ord = 2)
		beta = r_norm2
		V_list = [np.zeros(N)] #Stores columns of V matrix
		V_list[0] = r.reshape(-1, order='F')/beta
		H_list = [np.zeros(m_Krylov)] #Stores rows of Hessenberg matrix
		
		for m in range(1,m_Krylov+1):
			st_iter = time.time()
			V_list.append(np.zeros(N)) #Reserving space for vector (column of V) v_{j+1}
			H_list.append(np.zeros(m_Krylov)) #Reserving space for row of H h_{j+1}
			st_arnoldi = time.time()
			m_res = Arnoldi(V_list, H_list, (m-1), m,  LinOp)
			if printout: print("Arnoldi time:", time.time()-st_arnoldi)
			V = (np.array(V_list[:m_res])).T #Slicing V_list[:m] because v_{m+1} is not needed for projection step.
			H = (np.array(H_list))[:,:m_res] #Slicing because everything right to m'th column is placeholding zeros.
			st_lsq = time.time()
			y = LSq(beta, H)
			if printout: print("LSq time:", time.time()-st_lsq)
			st_proj = time.time()
			x = x_0 + V@y
			if printout: print("Projection step time:", time.time()-st_proj)
			r_norm2_inner = np.linalg.norm(b-LinOp(x), ord = 2)
			relres_inner = r_norm2_inner/np.linalg.norm(b, ord = 2)
			iterdata(relres_inner, printout) #rel residual
			if (relres_inner < eps):
				break_outer = True
				break
			et_iter = time.time()
			if printout: print(f"m = {m}; Absolute residual = :", r_norm2_inner, f"; Iteration time: {et_iter-st_iter} s")
		x_0 = x
		et_restart = time.time()
		if printout: print("Restart time:", et_restart - st_restart)
	et = time.time()
	elapsed = et - st
	if printout: print(f"Average iteration time: {elapsed/iterdata.iterations[-1]} s")
	if printout: print(f"GMRES(m) time: {elapsed} s")
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return x, solutiondata
#---

#---GMRES SciPy ver---

def GMRES_scipy(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13, printout = False):
	N = x_0.shape[0] #N=n^2
	iterdata = iterations_data()
	r = b - LinOp(x_0) #initial residual
	iterdata(np.linalg.norm(r, ord = 2)/np.linalg.norm(b, ord = 2), printout)
	G = scsp.linalg.LinearOperator((N,N), matvec = LinOp)
	st = time.time()
	s, data = scsp.linalg.gmres(G, b, x0=x_0, atol=eps, restart=m_Krylov, maxiter=None, M=None, callback=iterdata, callback_type=None)
	et = time.time()
	elapsed = et - st
	if printout: print("Average iteration time:", elapsed/iterdata.iterations[-1])
	if printout: print("Elapsed:", elapsed)
	if printout: print("Solution:", s)
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return s, solutiondata
#---

#--- CGNR ---

def CGNR(LinOp, LinOp_conj, x_0, b, inner = lambda u,v : v.T@u , maxiter=10000, eps = 1e-13, printout = False):
	x = x_0
	r = b - LinOp(x)
	z = LinOp_conj(r)
	p = z
	for k in range(maxiter):
		w = LinOp(p)
		alpha = inner(z,z)/inner(w,w)
		x = x + alpha*p
		r = r - alpha*w
		r_norm2 = np.linalg.norm( r, ord = 'fro')
		z_kp1 = LinOp_conj(r)
		if printout: print(f"iteration {k} ; ||r||_2 = {r_norm2}") 
		if r_norm2 < eps:
			break
		beta = inner(z_kp1,z_kp1)/inner(z,z)
		p = z_kp1 + beta*p
		z = z_kp1
	return x
#---

#--- Alternating optimization solver (proposed by German Z. Alekhin, MSU CMC)---

class simrank_ops:
	def __init__(self, A, c):
		self.c = c
		self.A = A
		self.n = A.shape[1]
		self.ATA = A.T@A
	def off(self, A):
		result = A.copy()
		np.fill_diagonal(result, 0)
		return result
	def mat_inner(self, A,B):
		return np.trace(B.T@A)
	def F_hat(self, Y):
		return Y - self.c*self.off(self.A.T@Y@self.A)
	def F_hat_conj(self, Y):
		return Y - self.c*self.A@self.off(Y)@self.A.T
	def F_UV(self, U, V):
		return self.c*self.off((self.A.T@U)@(V.T@self.A))+self.c*self.off(self.ATA)

	def F_hat_U(self, Y, U):
		return U@Y - self.c*self.off((self.A.T@U)@(Y@self.A))
	def F_hat_U_conj(self, Y, U):
		return U.T@Y - self.c*( (U.T@self.A)@(self.off(Y)@self.A.T) )

	def F_hat_Vt(self, Y, Vt):
		return Y@Vt - self.c*self.off((self.A.T@Y)@(Vt@self.A))
	def F_hat_Vt_conj(self, Y, Vt):
		return Y@Vt.T-self.c*( (self.A@self.off(Y))@(self.A.T@Vt.T) )

def ALS(U, V, A, c, dir_maxit, printout): #Alternating least squares solver
	#NOTE: computation of pinv as (A.TA)^-1 A.T is faster but numerically unstable
	simrank_operators = simrank_ops(A, c)
	F = simrank_operators.F_UV
	for iter_v in range(dir_maxit):
		if printout: print(f"V direction iteration {iter_v}")
		V = ( np.linalg.pinv(U)@F(U,V) ).T
		#V = ( ( np.linalg.inv(U.T@U) )@U.T@F(U,V) ).T
	for iter_u in range(dir_maxit):
		if printout: print(f"U direction iteration {iter_u}")
		U = F(U,V)@np.linalg.pinv(V.T)
		#U = F(U,V)@( V@np.linalg.inv(V.T@V) )
	return U,V


def ANE(U, V, A, c, dir_maxit, printout): #Alternating normal equations
	#NOTE: initial guess in CGNR is important here! zeros gives much better solution than taking solution from prev global step
	if printout: print(f"Starting ANE solver (CGNR)...")
	Vt = V.T
	simrank_operators = simrank_ops(A, c)
	rhs = c*simrank_operators.off(A.T@A)
	if printout: print(f"V direction iterations (U fixed) :")
	mv = lambda Y: simrank_operators.F_hat_U(Y, U)
	mvconj = lambda Y: simrank_operators.F_hat_U_conj(Y, U)
	Vt = CGNR(mv, mvconj, np.zeros((Vt.shape[0],Vt.shape[1])), rhs, simrank_operators.mat_inner, dir_maxit, printout=True)
	
	if printout: print(f"U direction iterations (V fixed) :")
	mv = lambda Y: simrank_operators.F_hat_Vt(Y, Vt)
	mvconj = lambda Y: simrank_operators.F_hat_Vt_conj(Y, Vt)
	U = CGNR(mv, mvconj, np.zeros((U.shape[0],U.shape[1])), rhs, simrank_operators.mat_inner, dir_maxit, printout=True)
	return U,Vt.T

def AltOpt(A, c, r, solver, maxiter=100, dir_maxit=100, eps_fro = 1e-15, eps_cheb = 1e-12, printout = False): #Main alternating optimiztion function.

	iterdata = iterations_data()
	n = A.shape[1]
	np.random.seed(42)
	U = np.random.randn(n,r)
	V = np.random.randn(n,r)
	
	#cATA = c*A.T@A
	#np.fill_diagonal(cATA, 0.0)
	#U, s, VH = np.linalg.svd(cATA, full_matrices = False)
	#U = ( U@np.sqrt(np.diag(s)) )[:,:r]
	#V = ( ( np.sqrt(np.diag(s))@VH )[:r,:] ).T
	
	st = time.time()
	
	for k in range(maxiter):
		if printout: print(f"Alternating optimization iteration {k}")
		V_prev = V
		U_prev = U
		U, V = solver(U, V, A, c, dir_maxit, False)
		diff =U@V.T - U_prev@V_prev.T
		err_fro = (np.linalg.norm(diff, ord = 'fro'))
		iterdata(err_fro)
		print(f"||U^(k+1)@V^(k+1).T - U^(k)@V^(k).T||_C at iter {k} = {np.max(np.abs(diff))}")
		if  err_fro < eps_fro:
			if printout: print("Converged by err Fro")
			break
		if np.max(np.abs(diff)) < eps_cheb:
			if printout: print("Converged by err Cheb")
			break
	elapsed = time.time() - st
	print(f"Elapsed {elapsed}")
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return np.eye(n)+U@V.T, solutiondata
