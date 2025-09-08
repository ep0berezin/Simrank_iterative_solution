import numpy as np
import simrank_main as sm
import matplotlib.pyplot as plt

class results:
	def __init__(self, taskname, ranks, solver):
		self.ranks = ranks
		self.taskname = taskname
		self.err_Frob_frank = []
		self.err_Cheb_frank = []
		self.solver = solver
	def getresults(self):
		for rank in ranks:
			args = {"acc":1e-5, "m_Krylov": 15, "rank": rank, "k_iter_max": 100, "taskname": self.taskname, "c": 0.8, "solvers":[self.solver, "Compare"], "showfig":0}
			err_Frob, err_Cheb = sm.launch(args)
			self.err_Frob_frank.append(err_Frob), self.err_Cheb_frank.append(err_Cheb)

if __name__=="__main__":
	taskname = "metro"
	ranks = np.arange(100, 200, 50)
	ALS_results = results(taskname, ranks, "AltOpt")
	#ALS_results.getresults()
	Newton_results = results(taskname, ranks, "Optimization_Newton")
	Newton_results.getresults()
	RSVD_results = results(taskname, ranks, "RSVIters")
	#RSVD_results.getresults()
	
	plt.figure()
	plt.grid()
	plt.plot(ranks, Newton_results.err_Cheb_frank, marker = '.')
	plt.xlabel(r"Approximation rank $r$")
	plt.ylabel(r"Chebyshev norm error, $||S-S_{approx}||_C$")
	plt.title("Cheb from rank")
	plt.show()
	

