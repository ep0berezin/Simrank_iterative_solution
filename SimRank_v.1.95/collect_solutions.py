import numpy as np
import simrank_main as sm
import datetime as dt

dateformat = "%Y-%m-%d-%H-%M-%S"
path = "results/solutions"


def getsolution(taskname, ranks, solvers):
	args = {
	"acc":1e-5,
	"m_Krylov": 15, 
	"rank": 200, 
	"k_iter_max": 100, 
	"taskname": taskname, 
	"c": 0.8, 
	"solvers": solvers,  
	"optimize" : True, 
	"showfig": False}
	#args = sm.load_args(sm.proc_args().argsfrom)
	for rank in ranks:
		args['rank'] = rank
		S, _, _ = sm.launch(args)
		solvers = args['solvers']
		for solver in solvers:
			np.save(
			f"{path}/S_{args['taskname']}_rank_{rank}_c_{args['c']}_solver_{solver}_{dt.datetime.now().strftime(dateformat)}.npy",
			S[list(S.keys())[0]],)

if __name__ == "__main__":
	ranks_metro = np.arange(10, 304, 10)
	ranks_eumail = np.arange(100, 1006, 100)
	ranks_fb = np.arange(100, 4097, 100)
	#getsolution("metro", ranks_metro, ["AltOpt"])

	getsolution("metro", ranks_metro, ["Optimization_Newton"])
	getsolution("metro", ranks_metro, ["RSVDIters"])

	getsolution("eumail", ranks_eumail, ["AltOpt"])
	getsolution("eumail", ranks_eumail, ["Optimization_Newton"])
	getsolution("eumail", ranks_eumail, ["RSVDIters"])
