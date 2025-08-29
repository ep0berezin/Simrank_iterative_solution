import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import scipy.sparse as scsp
import sys
import time
from memory_profiler import profile


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
	y = np.linalg.inv(H.T@H)@H.T@b
	return y

def Arnoldi(V_list, h_list, m_start, m_Krylov, LinOp, eps_zero = 1e-15, printout = False):
	for j in range(m_start, m_Krylov):
		#print(f"Building Arnoldi: V[:,{j}]")
		v_j = V_list[j]
		st_1 = time.time()
		w_j = colvecto1dim(LinOp(v_j))
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
		V_list[j+1] = (w_j/w_j_norm2)
	return m_Krylov

def colvecto1dim(u):
	tmp = u
	return tmp.reshape(tmp.shape[0], order = 'F')

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
		V_list[0] = colvecto1dim(r)/beta
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
	simrank_operators = simrank_ops(A, c)
	F = simrank_operators.F_UV
	for iter_v in range(dir_maxit):
		if printout: print(f"V direction iteration {iter_v}")
		#V = (np.linalg.lstsq(U, F(U,V), rcond= None)[0] ).T
		V = ( np.linalg.pinv(U)@F(U,V) ).T
		#V = ( ( np.linalg.inv(U.T@U) )@U.T@F(U,V) ).T
	for iter_u in range(dir_maxit):
		if printout: print(f"U direction iteration {iter_u}")
		#U = np.linalg.lstsq(V, F(U,V).T, rcond = None)[0].T
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
		U, V = solver(U, V, A, c, dir_maxit, True)
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

#---Khranilische khlama a.k.a Crap storage---
'''

class F_operator:
	def __init__(self, A, c):
		self.A = A
		self.n = A.shape[1]
		self.c = c
	def __call__(self, U):
		UA = U@self.A
		T_1 = self.c*(UA).T@UA
		np.fill_diagonal(T_1, 0.0) #cA.T@U.T@U@A - c diag
		T_2 = self.c*self.A.T@self.A
		np.fill_diagonal(T_2, 0.0) #cA.T@A - c diag
		res = T_1+T_2 - U.T@U
		return [res, UA] #UA needed to use it in grad(f) calculation.
class normal_op_vec:
	def __init__(self, X, d1, d2):
		self.XTX = X.T@X
		self.d1, self.d2 = d1, d2
	def __call__(self, vec):
		return (self.XTX@vec.reshape((self.d2,self.d1), order = 'F')).reshape((self.d2*self.d1,1), order = 'F')

def AGMRES(F, U, V, maxiter, printout, eps=1e-13): #alternating gmres
	n, r = U.shape
	VTV_op = normal_op_vec(V, n, r)
	uT, _ = GMRES_m(VTV_op, 15, U.reshape((n*r,1), order = 'F'), (V.T@(F(U,V).T)).reshape((r*n,1), order = 'F'), maxiter, eps)
	U = uT.reshape((r, n), order = 'F').T
	UTU_op = normal_op_vec(U, n, r)
	vT, _ = GMRES_m(UTU_op, 15, V.reshape((n*r,1), order = 'F'), (U.T@F(U,V)).reshape((r*n,1), order = 'F'), maxiter, eps)
	V = vT.reshape((r, n), order = 'F').T
	return U,V
	
	
class F_diag_operator:
	def __init__(self, A, c):
		self.c = c
		self.A = A
		self.n = A.shape[1]
		self.ATA = A.T@A
	def off(self, A):
		np.fill_diagonal(A, 0.0)
		return A
	def __call__(self, U,V):
		res = self.c*self.off(self.ATA)+self.c*self.off((self.A.T@U)@(V.T@self.A)-self.A.T@np.diag(np.diag(U@V.T))@self.A)+np.diag(np.diag(U@V.T))
		return res


def Oddsolver_AGMRES(A, c, r, maxiter, dir_maxit=10, eps_fro = 1e-5, eps_cheb = 1e-2, printout = True):
	F = F_operator(A, c)
	iterdata = iterations_data()
	n = A.shape[1]
	np.random.seed(42)
	U = np.random.randn(n,r)
	V = np.random.randn(n,r)
	st = time.time()
	err_fro = (np.linalg.norm(U@V.T, ord = 'fro')) #initial iteration
	iterdata(err_fro)
	for k in range(maxiter):
		if printout: print(f"General iteration {k}")
		V_prev = V
		U_prev = U
		U, V = AGMRES(F, U, V, printout=printout)
		diff =U@V.T - U_prev@V_prev.T
		err_fro = (np.linalg.norm(diff, ord = 'fro'))
		iterdata(err_fro)
		if  err_fro < eps_fro:
			if printout: print("Converged by err Fro")
			break
		if np.max(np.abs(diff)) < eps_cheb:
			if printout: print("Converged by err Cheb")
			break
	elapsed = time.time() - st
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return np.eye(n)+U@V.T, solutiondata

#---Optimization Newton---
#Solves grad(f) = 0 where f = ||F(U)||_F^2, F(U) = cA.T@U.T@A - c diag(A.T@U.T@U@A) +cA.T@A-c diag(A.T@A) - U.T@U

def dF_du(F_U, U, UA, A, c):
	#F_U precalulated
	n = A.shape[1]
	dFdu = np.zeros((n,n))
	#F_U, UA = F(U)
	delta = lambda i, j : 1.0 if (i==j) else 0.0
	tmp = 0.
	for p in range(n):
		for q in range(n):
			for i in range(n):
				for j in range(n):
					tmp += F_U[i,j]*( c*(1-delta(i,j))*(A[q,i]*UA[p,j]+A[q,j]*UA[p,i]) - (delta(i,q)*U[p,j] + delta(j,q)*U[p,i]) )
			dFdu[p,q] = tmp
	return dFdu

def grad_f(F_U, A, dFdu, c):
	#F_U precalulated
	n = A.shape[1]
	nabla_f = np.zeros((n,n))
	#F_U, UA = F(U)
	delta = lambda i, j : 1.0 if (i==j) else 0.0
	tmp = 0.
	for p in range(n):
		for q in range(n):
			for i in range(n):
				for j in range(n):
					#tmp += F_U[i,j]*( c*(1-delta(i,j))*(A[q,i]*UA[p,j]+A[q,j]*UA[p,i]) - (delta(i,q)*U[p,j] + delta(j,q)*U[p,i]) )
					tmp += F_U[i,j]*dFdu[p,q]
			nabla_f[p,q] = 2.*tmp
	return nabla_f #1d array filled with df/du_pq

def Jacobian(F_U, A, dFdu, c):
	n = A.shape[1]
	N = n**2
	J = np.zeros((N,N))
	tmp = 0.
	delta = lambda i, j : 1.0 if (i==j) else 0.0
	for p in range(n):
		for q in range(n):
			for alpha in range(n):
				for beta in range(n):
					for i in range(n):
						for j in range(n):
							tmp += dFdu[alpha,beta]*dFdu[p,q]+F_U[i,j]*c*(A[q,i]*A[beta,j]+A[q,j]*A[beta,i])*(1-delta(i,j))-delta(alpha,p)*(delta(i,q)*delta(beta,j)+delta(j,q)*delta(beta,i))
					J[n*p+q,n*alpha+beta] = 2.*tmp
	print("Jacobian")
	print(J)
	return J
	
def Opti_Newton(A, c, eps, k_iter_max):
	iterdata = iterations_data()
	n = A.shape[1]
	N = n**2
	F = F_operator(A,c)
	u_k = F(np.zeros((n,n)))[0].reshape((N), order = 'F')
	st = time.time()
	for k in range(k_iter_max):
		print(f"Iteration {k}")
		U_k = u_k.reshape((n,n), order ='F')
		[F_U_k, U_kA] = F(U_k)
		print("Calculating dFdu")
		dFdu = dF_du(F_U_k, U_k, U_kA, A, c)
		print("Calculating grad(f)")
		grad = grad_f(F_U_k, A, dFdu, c)
		print("performing Newton step")
		u_kp1 = u_k - np.linalg.pinv(Jacobian(F_U_k,A, dFdu, c))@grad.reshape((N)) #Note: reshape without fortran-like order. because grad stored as [ df du_11 ... df du_1n ; df du_21 ... df_du_2n ; ... ] so row-wise reshape.
		residual = np.linalg.norm(u_kp1-u_k, ord = 2)
		iterdata(residual)
		if residual < eps:
			break
		u_k = u_kp1
	elapsed = time.time() - st
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return u_kp1, solutiondata

#---

#---MinRes---
def MinRes(k_max, A, c, N, eps = 1e-13): # OUTDATED/rework to call in GMRES_m similar manner.
	I = np.identity(N)
	I_vec = I.reshape((N**2, 1), order = 'F')
	st = time.time()
	s = I_vec
	residuals = []
	iterations = []
	
	r = I_vec - G(s, A, c, N)
	p = G(r, A, c, N)
	
	residual = np.linalg.norm(r, ord = 2)
	residuals.append(residual)
	iterations.append(0)
	print(f"Iteration: {0}")
	print(f"Residual: {residual}")
	
	for k in range(1,k_max):
		a = (r.T@r)/(p.T@p)
		s = s + a*r
		r = r - a*p
		p = G(r, A, c, N)
		iterations.append(k)
		#print(S)
		residual = np.linalg.norm(r, ord = 2)
		residuals.append(residual)
		print(f"Iteration: {k}")
		print(f"Residual: {residual}")
		if (residual < eps):
			break
	et = time.time()
	elapsed = et - st
	print(f"Elapsed: {elapsed} s")
	#plt.figure()
	#plt.grid()
	res_graph = plt.plot(iterations, residuals, color = 'red')
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)
	S = s.reshape((N,N), order = 'F')
	return S
#---
'''
