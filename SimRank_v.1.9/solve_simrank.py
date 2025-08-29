import numpy as np
from scipy.sparse import csr_matrix
import sys
import time
import matplotlib.pyplot as plt 
import datetime as dt
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter
from memory_profiler import profile
import solvers as slv
import opti_solvers as optslv
import rsvd_solver as rsvdslv
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

def plot(solvername, taskname, solutiondata, acc, c, wtf=True):
	dateformat = "%Y_%m_%d-%H-%M-%S" #for log and plots saving
	res_graph = plt.plot(solutiondata[0], solutiondata[1])
	plt.yscale('log')
	plt.xlabel(r'Итерация', fontsize = 12) 
	plt.ylabel(r'Относительная невязка', fontsize = 12)
	
	if wtf:
		df = pd.DataFrame({'residuals': solutiondata[1], 'iterations': solutiondata[0]})
		df.to_csv("results/data/results_"+solvername+"_"+taskname+"_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat))+".csv", index=False)

def writelog(c, taskname, t, acc, dateformat):
	filename = ("results/log/log_"+taskname+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".csv")
	with open(filename, 'a+') as f:
		f.write(f"{c},{taskname},")
		for key in t:
			f.write(f"{key},{t[key]},")

def writerrs(taskname, solver, err_fro, err_cheb, rank):
	filename = ("results/log_"+solver+"_"+taskname+".csv")
	with open(filename, 'a+') as f:
		f.write(f"{taskname},{rank},{err_cheb},{err_fro},\n")

def err(S, n): #S - dict of solutions ###rework to make output like : d = {"gmres-gmres_svd" : [max_err, avg_err, fro_err]}
	max_err = 0.0
	err_tmp = 0.0
	err_sum = 0.0
	for key1 in S:
		for key2 in S:
			if (key1!=key2):

				plt.figure()
				graph_err = plt.imshow(np.abs(S[key1]-S[key2])) #errors portrait
				cbar = plt.colorbar()
				cbar.set_label("abs error")
				plt.title(f"Портрет ошибки {str(key1)} - {str(key2)}", fontweight = "bold")
				err_tmp = np.max(np.abs(S[key1]-S[key2]))
				err_sum += np.sum(np.abs(S[key1]-S[key2]))
				err_frob = np.linalg.norm((S[key1] - S[key2]), ord = 'fro')
			if (err_tmp > max_err):
				max_err = err_tmp
	return max_err, err_sum/(n*n), err_frob

def Solve(acc, m_Krylov, rank, k_iter_max, taskname, A, c, solvers): #solvers = list of flags: ['SimpleIter, GMRES, MinRes'] (in any order)
	dateformat = "%Y_%m_%d-%H-%M-%S" #for log and plots saving
	n = A.shape[0]
	if (A.shape[0]!=A.shape[1]):
		print("Non-square matrix passed in argument. Stopped.")
		return 1
	I = np.eye(n) #identity matrix of required dimensions
	I_vec = np.eye(n).reshape((n**2,1), order = 'F')
	print("Adjacency matrix:")
	print(A)
	A_csr = csr_matrix(A) #if A is already CSR -> changes nothing.
	
	S = {} #init dict of solutions
	t = {} #init time dict
	notfound = True
	plt.figure()
	plt.grid()
	for solver in solvers:
		if (solver == "SimpleIter"): #classis simple iter
			notfound = False
			print(f"Starting SimpleIter with {k_iter_max} iterations limit tau =  {tau} iter parameter ..")
			G = G_operator(A_csr, c)
			ts = time.time()
			tau = 1.
			s_si, solutiondata = slv.SimpleIter(G, tau, np.zeros((n,n)).reshape((n**2,1), order = 'F'), I_vec, k_iter_max, False, acc)
			ts = time.time() - ts
			S_si = s_si.reshape((n,n), order = 'F')
			#np.save(f"S_etalon_{taskname}.npy", S_si)
			plot(solver, taskname, solutiondata, acc, c)
			S["S_si"] = S_si
			t["S_si"] = ts
		if (solver == "GMRES"):
			notfound = False
			print(f"Starting GMRES with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			G = G_operator(A_csr, c)
			ts = time.time()
			s_gmres, solutiondata = slv.GMRES_m(G, m_Krylov, np.zeros((n,n)).reshape((n**2,1), order = 'F'), I_vec, k_iter_max, acc)
			ts = time.time() - ts
			S_gmres = s_gmres.reshape((n,n), order = 'F')
			plot(solver, taskname, solutiondata, acc, c)
			S["S_gmres"] = S_gmres
			t["S_gmres"] = ts
		if (solver == "GMRES_scipy"): 
			notfound = False
			print(f"Starting GMRES from SciPy with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ..")
			G = G_operator(A_csr, c)
			ts = time.time()
			s_gmres_scipy, solutiondata = slv.GMRES_scipy(G, m_Krylov, np.zeros((n,n)).reshape((n**2,1), order = 'F'), b, k_iter_max, acc)
			S_gmres_scipy = s_gmres_scipy.reshape((n,n), order = 'F')
			ts = time.time() - ts
			plot(solver, taskname, solutiondata, acc, c)
			S["S_gmres_scipy"] = S_gmres_scipy
			t["S_gmres_scipy"] = ts
		if (solver == "AltOpt"):
			notfound = False
			print(f"Starting oddsolver with {k_iter_max} iterations limit  ..")
			ts = time.time()
			rank = 200
			S_odd, solutiondata = slv.AltOpt(A, c, rank, slv.ALS, 100, 50, printout = True)
			print(S_odd.shape)
			ts = time.time() - ts
			plot(solver, taskname, solutiondata, acc, c)
			S["S_odd"] = S_odd
			t["S_odd"] = ts
		if (solver == "Optimization_Newton"):
			notfound = False
			print(f"Starting optimization Newton with {k_iter_max} iterations  ..")
			#NOTE: best results are obtained if gmres_restarts=1 and m_Krylov 10...20 used.
			#Increasing m_Krylov (and generally total amount of iterations, i.e. gmres_restarts*m_Krylov) over 20 leads to worse results.
			ts = time.time()
			rank = 200
			S_opti_newton, solutiondata = optslv.Newton(A, c, rank, maxiter=100, gmres_restarts=1, m_Krylov=15, solver=optslv.GMRES_scipy)
			ts = time.time() - ts
			plot(solver, taskname, solutiondata, acc, c)
			S["S_opti_newton"] = S_opti_newton
			t["S_opti_newton"] = ts
		if (solver == "RSVDIters"):
			notfound = False
			print(f"Starting RSVD Iters with {k_iter_max} iterations limit tau =  {tau} iter parameter ..")
			ts = time.time()
			r = 500
			M_rsvd, solutiondata =  rsvdslv.RSVDIters(A, c, r, p, k_iter_max, acc*1e4) #1e-5 too slow.
			ts = time.time() - ts
			plot(solver, taskname, solutiondata, acc, c)
			S["S_rsvd"] = M_rsvd + I
			t["S_rsvd"] = ts
		if (solver == "SimrankNX"): #test for simrank nx.
			notfound = False
			print(f"Starting SimpleIter NX with {k_iter_max} iterations limit  ..")
			A_r = np.where(A_csr.toarray()>0, 1, 0)
			print("Restored adjacency matrix A_r:")
			print(A_r)
			Graph = nx.from_numpy_array(A_r, create_using=nx.MultiDiGraph())
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
		if (solver == "Compare"):
			notfound = False
			print(f"Comparing with etalon {taskname}")
			S_cmp = np.load(f"data/S_etalon_{taskname}.npy")
			#rank_S = np.linalg.matrix_rank(S_cmp)
			#print(f"rank(S_{taskname}) = {rank_S}")
			S["S_cmp"] = S_cmp
			t["S_cmp"] = 0.0
		if notfound:
			print("Solver not found.")
			return 1
	plt.savefig("results/img/vis_"+taskname+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
	writelog(c, taskname, t, acc, dateformat)
	thresholds = [0.2, 0.5]
	if len(S)>1:
		err_Cheb, avg_err, err_Frob = err(S, n)
		print(f"Err Frobenius = {err_Frob}")
		print(f"Err Frobenius rel= {err_Frob/np.linalg.norm(np.load(f"data/S_etalon_{taskname}.npy"), ord = 'fro')}") ####!!!
		print(f"Max err (Chebyshev) = {err_Cheb}")
		print(f"Avg err ( sum(|S_1 - S_2|) / n*n) = {avg_err}")
		#writerrs(taskname, "RSVD", err_Frob, err_Cheb, r) 
	for key in S: #iterating over solutions
		print(key)
		print(S[key])
		print(f"Time of {key}: ", t[key])
		plt.figure()
		graph_log = plt.imshow(np.log(S[key]-I+1e-1)) #1e-1 = small delta to get rid of negative values
		cbar = plt.colorbar()
		cbar.set_label("ln(S[i,j])")
		plt.title(taskname)
		plt.title(f"Матрица S, логарифмическая шкала, метод: {str(key)}", fontweight = "bold")
		plt.savefig("results/img/imshow_ln_"+taskname+"_"+str(key)+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
		
		plt.figure()
		graph = plt.imshow(S[key]-I)
		cbar = plt.colorbar()
		cbar.set_label("(S[i,j])")
		plt.title(taskname)
		plt.title(f"Матрица S, метод: {str(key)}", fontweight = "bold")
		plt.savefig("results/img/imshow_"+taskname+"_"+str(key)+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
		for trs in thresholds: #printing thresholded matrices
			plt.figure()
			plt.grid()
			graph = plt.imshow(np.where((S[key]-I)>=trs, 1.0, 0.0), cmap = "binary")
			x_dots, y_dots = np.where((S[key]-I)>=trs)
			vals = S[key][y_dots, x_dots]
			plt.scatter(x_dots, y_dots, c = vals, cmap = "viridis", s=5)
			cbar = plt.colorbar()
			cbar.set_label("(S[i,j])")
			plt.title(taskname)
			plt.title(f"Матрица S, метод: {str(key)}, порог: {str(trs)}", fontweight = "bold")
			plt.savefig("results/img/imshow_top_"+str(trs)+"_"+taskname+"_"+str(key)+"_eps_"+str(acc)+"_c_"+str(c)+"_"+str(dt.datetime.now().strftime(dateformat) )+".png")
	plt.show()
	return S


