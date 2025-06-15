import numpy as np
from scipy.sparse import csr_matrix
import sys
import time
import matplotlib.pyplot as plt 
import datetime as dt
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile
import solvers as slv
import networkx as nx



class G_operator:
	def __init__(self, A, c):
		self.A = A
		self.n = self.A.shape[1]
		if (self.n!=self.A.shape[0]):
			print(f"Warning! Non-square adjacency matrix detected when constructing operator.")
		self.c = c
	def __call__(self, u):
		U = u.reshape((self.n, self.n), order = 'F')
		T_1 = self.A.T@U@self.A
		np.fill_diagonal(T_1, 0.0) #A.TUA - diag(A.TUA)
		G = U - self.c*T_1
		G = G.reshape((self.n**2,1), order = 'F')
		return G

class G_svd_operator:
	def __init__(self, A, c, sparse_svd_factors = False, svd_trs=1e-5, r_coef=2.0):
		#about r_coef. Analytically, r_coef must be 1/2, but 4.0 works fine and gives speedup, __in case of eumail__. 
		#Ofc because operator singvals and singvecs contains small values that are being thresholded but ...
		self.n = A.shape[1]
		nnz = A.nnz
		print(f"A shape: {A.shape}, nnz: {nnz}")
		if (self.n!=A.shape[0]):
			print(f"Warning! Non-square adjacency matrix detected when constructing operator.")
		self.c = c
		r = int( (nnz/self.n)*r_coef ) #obtained from condition: 4rn^2 < 2 n_{nnz}*n -- condition of SVD-version efficiency relatively to CSR-version.
		print(f"Truncating rank: {r}")
		print("SVD of adjacency matrix...")
		U, s, V = np.linalg.svd(A.toarray(), full_matrices = False)
		self.W_r = U[:,:r]@np.diag(s[:r])
		self.W_r_T = self.W_r.T
		self.V_r = V[:r,:]
		self.V_r_T = self.V_r.T
		
		if sparse_svd_factors:
			print("Completed. Thresholding...")
			for i in range(U.shape[0]):
				for j in range(U.shape[1]):
					if (np.abs(U[i,j])<svd_trs):
						U[i,j] = 0
			for i in range(V.shape[0]):
				for j in range(V.shape[1]):
					if (np.abs(V[i,j])<svd_trs):
						V[i,j] = 0
			for i in range(s.shape[0]): #not sure it is needed since we truncate by rank anyway; but maybe it can sometimes cut low singular vals.
				if (np.abs(s[i])<svd_trs):
						s[i] = 0
			print("Thresholding completed.")
			print("U: ")
			print(U)
			print("sigma: ")
			print(s)
			print("Transforming to csr...")
			W_r = U[:,:r]@np.diag(s[:r])
			self.W_r = csr_matrix(W_r)
			print(f"SVD factors sparsity defined as sparsity = 1 - nnz/(dim1*dim2):")
			print(f"W = U@Sigma sparsity: {1 - self.W_r.nnz/(self.W_r.shape[0]*self.W_r.shape[1])}")
			self.V_r = csr_matrix(V[:r])
			print(f"V sparsity: {1 - self.V_r.nnz/(self.V_r.shape[0]*self.V_r.shape[1])}")
			self.W_r_T = csr_matrix(W_r.T)
			self.V_r_T = csr_matrix(V[:r].T)
			print("Transformed.")
	def __call__(self, x):
		X = x.reshape((self.n, self.n), order = 'F')
		T_0 = self.V_r_T@(self.W_r_T@X) # 2rn^2
		T_1 = (T_0@self.W_r)@self.V_r # 2rn^2 totally 4 rn^2
		np.fill_diagonal(T_1, 0.0) #A.TUA - diag(A.TUA)
		G = X - self.c*T_1
		G = G.reshape((self.n**2,1), order = 'F')
		return G

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
	
def plot(solvername, taskname, solutiondata, acc):
	res_graph = plt.plot(solutiondata[0], solutiondata[1])
	plt.yscale('log')
	plt.xlabel(r'Итерация', fontsize = 12) 
	plt.ylabel(r'Относительная невязка', fontsize = 12)

def writelog(c, taskname, t, acc, dateformat):
	filename = ("results/log_"+taskname+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".csv")
	with open(filename, 'a+') as f:
		f.write(f"{c},{taskname},")
		for key in t:
			f.write(f"{key},{t[key]},")

def maxerr(S): #S - dict of solutions
	err = 0.0
	err_tmp = 0.0
	for key1 in S:
		for key2 in S:
			if (key1!=key2):
				plt.figure()
				graph_err = plt.imshow(np.abs(S[key1]-S[key2])) #errors portrait
				cbar = plt.colorbar()
				cbar.set_label("abs error")
				plt.title(f"Портрет ошибки {str(key1)} - {str(key2)}", fontweight = "bold")
			err_tmp = np.max(np.abs(S[key1]-S[key2]))
			if (err_tmp > err):
				err = err_tmp

	return err

def RSVDIters(A, c, r, p, k_max_iter, eps): #RSVD iterations.
	n = A.shape[0]
	iterdata = slv.iterations_data()
	F_M = F_M_operator(A,c) #init operator
	st = time.time()
	M_0 = np.zeros((n,n))
	M_prev = F_M(M_0) #obtain M_1 = B
	U, sigma, V = np.linalg.svd(M_prev, full_matrices = False)
	U_hat = U*np.sqrt(sigma) #Get U_hat = U@sqrt(Sigma); store this way to effectively compute diag()*w_i.
	
	for k in range(k_max_iter):
		MOmega = F_M.randomize(U_hat, r, p) #effective multiplication by Omega
		Q, R = np.linalg.qr(MOmega) #QR decompo
		t_QTM = Q.T@M_prev #obtain Q^T @ M
		M_r = Q@t_QTM #M_r
		U_0, sigma, V = np.linalg.svd(t_QTM, full_matrices = False) #small SVD
		U = (Q@U_0)[:,:r] #get U and truncate to r
		U_hat = U*np.sqrt(sigma[:r]) #saving U_hat
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
	

def Solve(acc, m_Krylov, tau, r_factor_rsvd, p, k_iter_max, taskname, A, c, solvers): #solvers = list of flags: ['SimpleIter, GMRES, MinRes'] (in any order)
	dateformat = "%Y_%m_%d-%H-%M-%S" #for log and plots saving
	n = A.shape[0]
	if (A.shape[0]!=A.shape[1]):
		print("Non-square matrix passed in argument. Stopped.")
		return 1
	I = np.identity(n) #identity matrix of required dimensions
	print("Adjacency matrix:")
	print(A)
	I = np.identity(n)
	I_vec = np.identity(n).reshape((n**2,1), order = 'F')
	#s_0 = I_vec
	s_0 = np.zeros((n,n)).reshape((n**2,1), order = 'F')
	b = I_vec
	
	A_csr = csr_matrix(A) #if A is already CSR -> changes nothing.
	
	S = {} #init dict of solutions
	t = {} #init time dict
	r = int(n*r_factor_rsvd) #RSVD rank
	notfound = True
	plt.figure()
	plt.grid()
	for solver in solvers:
		if (solver == "SimpleIter"): #classis simple iter
			notfound = False
			print(f"Starting SimpleIter with {k_iter_max} iterations limit tau =  {tau} iter parameter ..")
			G = G_operator(A_csr, c) #Initialize operator
			ts = time.time()
			s_si, solutiondata = slv.SimpleIter(G, tau, s_0, b, k_iter_max, acc)
			ts = time.time() - ts
			S_si = s_si.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_si"] = S_si
			t["S_si"] = ts
		if (solver == "SimpleIter_TEST"): #test SI with SVD for adjustable max iter
			notfound = False
			print(f"Starting SimpleIter with {k_iter_max} iterations limit tau =  {tau} iter parameter ..")
			G = G_operator(A_csr, c) #Initialize operator
			ts = time.time()
			s_si_TEST, solutiondata = slv.SimpleIter(G, tau, s_0, b, 77, acc)
			ts = time.time() - ts
			S_si_TEST = s_si_TEST.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_si_TEST"] = S_si_TEST
			t["S_si_TEST"] = ts
		if (solver == "SimpleIter_SVD"): #simple iter with adjacency matrix as sparse SVD.
			notfound = False
			print(f"Starting SimpleIterSVD with {k_iter_max} iterations limit tau =  {tau} iter parameter ..")
			G = G_svd_operator(A_csr, c, sparse_svd_factors = True) #Initialize operator
			ts = time.time()
			s_si_svd, solutiondata = slv.SimpleIter(G, tau, s_0, b, k_iter_max, acc)
			ts = time.time() - ts
			S_si_svd = s_si_svd.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_si_svd"] = S_si_svd
			t["S_si_svd"] = ts
		if (solver == "RSVDIters"):
			notfound = False
			print(f"Starting RSVD Iters with {k_iter_max} iterations limit tau =  {tau} iter parameter ..")
			ts = time.time()
			M_rsvd, solutiondata =  RSVDIters(A, c, r, p, k_iter_max, acc*1e4) #1e-5 too slow.
			ts = time.time() - ts
			plot(solver, taskname, solutiondata, acc)
			S["S_rsvd"] = M_rsvd + I
			t["S_rsvd"] = ts
		###NX
		if (solver == "SimrankNX"): #test for simrank nx.
			notfound = False
			print(f"Starting SimpleIter NX with {k_iter_max} iterations limit  ..")
			A_r = np.where(A_csr.toarray()>0, 1, 0)
			print("Restored adjacency matrix A_r:")
			print(A_r)
			Graph = nx.from_numpy_matrix(A_r, create_using=nx.MultiDiGraph())
			#plt.figure()
			#nx.draw(Graph)
			#plt.show()
			Graph.remove_edges_from(nx.selfloop_edges(Graph)) #remove loops
			ts = time.time()
			S_nx = nx.simrank_similarity(Graph, importance_factor=c, max_iterations = k_iter_max, tolerance = acc)
			ts = time.time() - ts
			print("Elapsed NX: ", ts)
			Snx_fin = np.zeros((n,n))
			for i in range(n):
				for j in range(n):
					Snx_fin[i,j] = S_nx[i][j]
			S["S_nx"] = Snx_fin
			t["S_nx"] = ts
		###
		if (solver == "GMRES"):
			notfound = False
			print(f"Starting GMRES with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			G = G_operator(A_csr, c) #Initialize operator
			ts = time.time()
			s_gmres, solutiondata = slv.GMRES_m(G, m_Krylov, s_0, b, k_iter_max, acc)
			ts = time.time() - ts
			S_gmres = s_gmres.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_gmres"] = S_gmres
			t["S_gmres"] = ts
		if (solver == "GMRES_SVD"):
			notfound = False
			print(f"Starting GMRES_SVD with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			G = G_svd_operator(A_csr, c, sparse_svd_factors = True) #Initialize operator
			ts = time.time()
			s_gmres_svd, solutiondata = slv.GMRES_m(G, m_Krylov, s_0, b, k_iter_max, acc)
			ts = time.time() - ts
			S_gmres_svd = s_gmres_svd.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_gmres_svd"] = S_gmres_svd
			t["S_gmres_svd"] = ts
		if (solver == "GMRES_scipy"):
			notfound = False
			print(f"Starting GMRES from SciPy with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			G = G_operator(A_csr, c) #Initialize operator
			ts = time.time()
			s_gmres_scipy, solutiondata = slv.GMRES_scipy(G, m_Krylov, s_0, b, k_iter_max, acc)
			S_gmres_scipy = s_gmres_scipy.reshape((n,n), order = 'F')
			ts = time.time() - ts
			plot(solver, taskname, solutiondata, acc)
			S["S_gmres_scipy"] = S_gmres_scipy
			t["S_gmres_scipy"] = ts
		if (solver == "MinRes"):
			notfound = False
			print(f"Starting MinRes with {k_iter_max} iterations limit  ...")
			S_minres = 0#placeholder
			plot(solver, taskname, solutiondata, acc)
			S["S_minres"] = S_minres
		if notfound:
			print("Solver not found.")
			return 1
	plt.savefig("results/vis_"+taskname+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
	writelog(c, taskname, t, acc, dateformat)
	thresholds = [0.2, 0.5]
	print("Max err:")
	print(maxerr(S))
	for key in S:
		print(key)
		print(S[key])
		print(f"Time of {key}: ", t[key])
		plt.figure()
		graph_log = plt.imshow(np.log(S[key]-I+1e-1)) #1e-1 = small delta to get rid of negative values
		cbar = plt.colorbar()
		cbar.set_label("ln(S[i,j])")
		plt.title(taskname)
		plt.title(f"Матрица S, логарифмическая шкала, метод: {str(key)}", fontweight = "bold")
		plt.savefig("results/imshow_ln_"+taskname+"_"+str(key)+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
		
		plt.figure()
		graph = plt.imshow(S[key]-I)
		cbar = plt.colorbar()
		cbar.set_label("(S[i,j])")
		plt.title(taskname)
		plt.title(f"Матрица S, метод: {str(key)}", fontweight = "bold")
		plt.savefig("results/imshow_"+taskname+"_"+str(key)+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
		for trs in thresholds:
			plt.figure()
			plt.grid()
			graph = plt.imshow(np.where((S[key]-I)>=trs, 1.0, 0.0), cmap = "binary")
			x_dots, y_dots = np.where((S[key]-I)>=trs)
			vals = S[key][y_dots, x_dots]
			plt.scatter(x_dots, y_dots, c = vals, cmap = "viridis")
			cbar = plt.colorbar()
			cbar.set_label("(S[i,j])")
			plt.title(taskname)
			plt.title(f"Матрица S, метод: {str(key)}, порог: {str(trs)}", fontweight = "bold")
			plt.savefig("results/imshow_top_"+str(trs)+"_"+taskname+"_"+str(key)+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
	plt.show()
	return S


