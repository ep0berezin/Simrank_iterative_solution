import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
print(f"Using threads: {os.environ.get('OMP_NUM_THREADS')}")
import importlib
import solve_simrank as solvesim

def main_process(args):
	d = {"acc":args[0], "m_Krylov": args[1], "rank": args[2], "k_max": args[3], "taskname": args[4], "c": args[5], "solvers":args[6]}
	#if (process_argv(sys.argv, d)):
	#	return
	print("Arguments:\n", d)
	acc = d["acc"]
	m_Krylov = d["m_Krylov"]
	rank = d["rank"]
	k_iter_max = d["k_max"]
	taskname = d["taskname"]
	c = d["c"]
	solvers = d["solvers"]
	dataset = importlib.import_module(taskname)
	A = dataset.ObtainMatrix()
	solvesim.Solve(acc, m_Krylov, rank, k_iter_max, taskname, A, c, solvers)
	

if __name__=="__main__":
	args = [1e-5,15, 200, 100, "fb", 0.8,  ["Optimization_Newton", "Compare"]] #default args
	main_process(args)
