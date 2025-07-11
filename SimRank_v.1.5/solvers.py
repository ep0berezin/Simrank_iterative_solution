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
	def __call__(self, r):
		print(f"Iteration: {self.k_iter}")
		self.k_iter+=1
		self.iterations.append(self.k_iter)
		self.residuals.append(r)
		print('Current residual =', r)
		return r

#---Fixed Point---
def SimpleIter(LinOp, tau, x_0, b, k_max, eps = 1e-13):
	N = x_0.shape[0] #N = n^2
	s = x_0
	iterdata = iterations_data()
	st = time.time()
	for k in range(k_max):
		s_prev = s
		s = (b-LinOp(s))*tau+s
		r_norm2 = np.linalg.norm(s-s_prev, ord = 2)
		relres = r_norm2/np.linalg.norm(b, ord = 2)
		iterdata(relres)
		if (relres  < eps):
			break
	et = time.time()
	elapsed = et - st
	print(f"Average iteration time: {elapsed/iterdata.iterations[-1]} s")
	print(f"Elapsed: {elapsed} s")
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

def Arnoldi(V_list, h_list, m_start, m_Krylov, LinOp, eps_zero = 1e-15):
	for j in range(m_start, m_Krylov):
		#print(f"Building Arnoldi: V[:,{j}]")
		v_j = V_list[j]
		st_1 = time.time()
		w_j = colvecto1dim(LinOp(v_j))
		print("Evaluate Av_j time:", time.time()-st_1)
		Av_j_norm2 = np.linalg.norm(w_j, ord = 2)
		st_2 = time.time()
		for i in range(j+1):
			v_i = V_list[i]
			h_ij = v_i@w_j  
			h_list[i][j] = h_ij
			w_j -= h_ij*v_i
		print("MGS for v_{j+1} time:", time.time()-st_2)
		w_j_norm2 = np.linalg.norm(w_j, ord = 2)
		h_list[j+1][j] = w_j_norm2
		if (w_j_norm2 <= eps_zero):
			return j
		V_list[j+1] = (w_j/w_j_norm2)
	return m_Krylov
	
'''
def Arnoldi_test(V_list, h_list, m_start, m_Krylov, LinOp, eps_zero = 1e-5):
	for j in range(m_start, m_Krylov):
		#print(f"Building Arnoldi: V[:,{j}]")
		w_j = colvecto1dim(LinOp(V_list[j]))
		Av_j_norm2 = np.linalg.norm(w_j, ord = 2)
		for i in range(j+1):
			h_list[i][j] = V_list[i]@w_j  
			w_j -= h_list[i][j]*V_list[i]
		w_j_norm2 = np.linalg.norm(w_j, ord = 2)
		h_list[j+1][j] = w_j_norm2
		if (w_j_norm2 <= eps_zero*Av_j_norm2):
			return j
		V_list[j+1] = (w_j/w_j_norm2)
	return m_Krylov
'''
def colvecto1dim(u):
	tmp = u
	return tmp.reshape(tmp.shape[0], order = 'F')

def GMRES_m(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13):
	N = x_0.shape[0] #N = n^2
	r = b - LinOp(x_0)
	r_norm2 = np.linalg.norm(r, ord = 2)
	relres = r_norm2/np.linalg.norm(b, ord = 2)
	iterdata = iterations_data()
	iterdata(relres)
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
			print("Arnoldi time:", time.time()-st_arnoldi)
			V = (np.array(V_list[:m_res])).T #Slicing V_list[:m] because v_{m+1} is not needed for projection step.
			H = (np.array(H_list))[:,:m_res] #Slicing because everything right to m'th column is placeholding zeros.
			st_lsq = time.time()
			y = LSq(beta, H)
			print("LSq time:", time.time()-st_lsq)
			st_proj = time.time()
			x = x_0 + V@y
			print("Projection step time:", time.time()-st_proj)
			r_norm2_inner = np.linalg.norm(b-LinOp(x), ord = 2)
			relres_inner = r_norm2_inner/np.linalg.norm(b, ord = 2)
			iterdata(relres_inner) #rel residual
			if (relres_inner < eps):
				break_outer = True
				break
			et_iter = time.time()
			print(f"m = {m}; Residual = :", r_norm2_inner, f"; Iteration time: {et_iter-st_iter} s")
		x_0 = x
		et_restart = time.time()
		print("Restart time:", et_restart - st_restart)
	et = time.time()
	elapsed = et - st
	print(f"Average iteration time: {elapsed/iterdata.iterations[-1]} s")
	print(f"GMRES(m) time: {elapsed} s")
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return x, solutiondata
#---

#---GMRES SciPy ver---


def GMRES_scipy(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13):
	N = x_0.shape[0] #N=n^2
	iterdata = iterations_data()
	r = b - LinOp(x_0) #initial residual
	iterdata(np.linalg.norm(r, ord = 2)/np.linalg.norm(b, ord = 2))
	G = scsp.linalg.LinearOperator((N,N), matvec = LinOp)
	st = time.time()
	s, data = scsp.linalg.gmres(G, b, x0=x_0, atol=eps, restart=m_Krylov, maxiter=None, M=None, callback=iterdata, callback_type=None)
	et = time.time()
	elapsed = et - st
	print("Average iteration time:", elapsed/iterdata.iterations[-1])
	print("Elapsed:", elapsed)
	print("Solution:", s)
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	return s, solutiondata
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
