import numpy as np
from scipy.sparse import csr_matrix
import sys
import time
import matplotlib.pyplot as plt 
import datetime as dt
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile
import solvers as slv

class G_operator:
	def __init__(self, A, c):
		self.A = A
		self.n = self.A.shape[1]
		if (self.n!=self.A.shape[0]):
			print(f"Warning! Non-zero adjacency matrix detected when constructing operator.")
		self.c = c
	def __call__(self, u):
		self.A = csr_matrix(self.A)
		U = u.reshape((self.n, self.n), order = 'F')
		ATUA = self.A.T@U@self.A
		G = U - self.c*ATUA+self.c*np.diag(np.diag(ATUA))
		G = G.reshape((self.n**2,1), order = 'F')
		return G

def plot(solvername, taskname, solutiondata, acc):
	res_graph = plt.plot(solutiondata[0], solutiondata[1])
	plt.yscale('log')
	plt.xlabel(r'$Iterations$', fontsize = 12) 
	plt.ylabel(r'$Local\quadresiduals$', fontsize = 12)

def Solve(acc, m_Krylov, tau, k_iter_max, taskname, A, c, solvers): #solvers = list of flags: ['SimpleIter, GMRES, MinRes'] (in any order)
	n = A.shape[0]
	if (A.shape[0]!=A.shape[1]):
		print("Non-square matrix passed in argument. Stopped.")
		return 1
	I = np.identity(n) #identity matrix of required dimensions
	print("Adjacency matrix:")
	print(A)
	I = np.identity(n)
	I_vec = np.identity(n).reshape((n**2,1), order = 'F')
	A_csr = csr_matrix(A) #if A is already CSR -> changes nothing.
	G = G_operator(A_csr, c) #Initialize operator
	S = {} #init dict of solutions
	notfound = True
	plt.figure()
	plt.grid()
	for solver in solvers:
		if (solver == "SimpleIter"):
			notfound = False
			s_si, solutiondata = slv.SimpleIter(G, tau, I_vec, I_vec, k_iter_max, acc)
			S_si = s_si.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_si"] = S_si
		if (solver == "GMRES"):
			notfound = False
			print(f"Starting GMRES with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			s_gmres, solutiondata = slv.GMRES_m(G, m_Krylov, I_vec, I_vec, k_iter_max, acc)
			S_gmres = s_gmres.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_gmres"] = S_gmres
		if (solver == "GMRES_scipy"):
			notfound = False
			print(f"Starting GMRES from SciPy with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			s_gmres_scipy, solutiondata = slv.GMRES_scipy(G, m_Krylov, I_vec, I_vec, k_iter_max, acc)
			S_gmres_scipy = s_gmres_scipy.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc)
			S["S_gmres_scipy"] = S_gmres_scipy
		if (solver == "MinRes"):
			notfound = False
			print(f"Starting MinRes with {k_iter_max} iterations limit  ...")
			S_minres = 0#placeholder
			plot(solver, taskname, solutiondata, acc)
			S["S_minres"] = S_minres
		if notfound:
			print("Solver not found.")
			return 1
	plt.savefig("results/_"+taskname+"_eps_"+str(acc)+"_"+str(dt.datetime.now())+".png")
	for key in S:
		print(key)
		plt.figure()
		graph = plt.imshow(np.log(S[key]-I+1e-15))
		cbar = plt.colorbar()
		cbar.set_label("ln(S[i,j])")
		plt.title(taskname)
		plt.title("Матрица S", fontweight = "bold")
	plt.show()
	return S


