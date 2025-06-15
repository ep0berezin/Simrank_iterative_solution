import sys
import fb
import solve_dlyap as sdlyap
import matplotlib.pyplot as plt 

def process_argv(argv, d):
	argc = len(argv)
	if (argc == 1):
		print("You have not used any flags. Use 'default' flag to load values from program;\n"\
		"Write in format:\n python3 *program*.py acc *value* m_Krylov *value* tau *value* k_max *value* taskname *task name* c *value* solvers *solver 1* *solver 2* ..."\
		"\nAvailable solvers: SimpleIter ; GMRES ; GMRES_scipy ; MinRes")
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
		return 0
	return 1

def main_process(args):
	d = {"acc":args[0], "m_Krylov": args[1], "tau": args[2], "k_max": args[3], "taskname": args[4], "c": args[5], "solvers":args[6]}
	if (process_argv(sys.argv, d)):
		return
	print("Arguments:\n", d)
	acc = d["acc"]
	m_Krylov = d["m_Krylov"]
	tau = d["tau"]
	k_iter_max = d["k_max"]
	taskname = d["taskname"]
	c = d["c"]
	solvers = d["solvers"]
	if (taskname == "Fb"):
		A = fb.ObtainMatrix()
	sdlyap.Solve(acc, m_Krylov, tau, k_iter_max, taskname, A, c, solvers)

if __name__=="__main__":
	args = [1e-5,15, 1, 1000000000, "Fb", 0.8,  ["GMRES"]] #default args
	main_process(args)
