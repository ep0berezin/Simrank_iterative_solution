import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
print("Using threads:")
print(os.environ.get("OMP_NUM_THREADS"))
import importlib
import solve_dlyap as sdlyap
import matplotlib.pyplot as plt 

def process_argv(argv, d):
	argc = len(argv)
	if (argc == 1):
		print("You have not used any flags. Use 'default' flag to load values from program;\n"\
		"Write in format:\n python3 *program*.py acc *value* m_Krylov *value* tau *value* k_max *value* taskname *task name* c *value* solvers *solver 1* *solver 2* ..."\
		"\nAvailable solvers: SimpleIter ; GMRES ; GMRES_scipy ; MinRes; lowrank")
		return 1
	if (argc>1 and argv[1] == "default"):
		return 0
	if (argc>2):
		for i_arg in range(argc):
			if (argv[i_arg] == "acc"):
				d["acc"] = float(argv[i_arg+1])
			if (argv[i_arg] == "m_Krylov"):
				d["m"] = int(argv[i_arg+1])
			if (argv[i_arg] == "tau"):
				d["tau"] = int(argv[i_arg+1])
			if (argv[i_arg] == "r_factor"):
				d["r_factor"] = int(argv[i_arg+1])
			if (argv[i_arg] == "p"):
				d["p"] = int(argv[i_arg+1])
			if (argv[i_arg] == "k_max"):
				d["k_max"] = int(argv[i_arg+1])
			if (argv[i_arg] == "taskname"):
				d["taskname"] = argv[i_arg+1]
			if (argv[i_arg] == "c"):
				d["c"] = float(argv[i_arg+1])
			if (argv[i_arg] == "solvers"):
				solvers_list = []
				for i in range(i_arg+1, argc):
					solvers_list.append(argv[i])
				d["solvers"] = solvers_list
		print(d["solvers"])
		return 0, 
	return 1

def main_process(args):
	d = {"acc":args[0], "m_Krylov": args[1], "tau": args[2], "r_factor": args[3], "p": args[4], "k_max": args[5], "taskname": args[6], "c": args[7], "solvers":args[8]}
	if (process_argv(sys.argv, d)):
		return
	print("Arguments:\n", d)
	acc = d["acc"]
	m_Krylov = d["m_Krylov"]
	tau = d["tau"]
	r_factor = d["r_factor"]
	p = d["p"]
	k_iter_max = d["k_max"]
	taskname = d["taskname"]
	c = d["c"]
	solvers = d["solvers"]
	dataset = importlib.import_module(taskname)
	A = dataset.ObtainMatrix()
	sdlyap.Solve(acc, m_Krylov, tau, r_factor, p, k_iter_max, taskname, A, c, solvers)
	

if __name__=="__main__":
	args = [1e-5,15, 1, 0.5, 8, 1000000000, "eumail", 0.8,  ["SimpleIter", "SimpleIter_SVD"]] #default args
	main_process(args)
