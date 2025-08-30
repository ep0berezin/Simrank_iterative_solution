import numpy as np
import scipy.sparse as scsp
import time
import solvers as slv

class iterations_data:
	def __init__(self):
		self.iterations = []
		self.funcvals = []
		self.k_iter = 0
	def __call__(self, f, printout = True):
		if printout: print(f"Iteration: {self.k_iter}")
		self.k_iter+=1
		self.iterations.append(self.k_iter)
		self.funcvals.append(f)
		if printout: print('Current f(U) value =', f)
		return f

class simrank_ops:
	def __init__(self, A, c, r):
		self.c = c
		self.A = A
		self.n = A.shape[1]
		self.ATA = A.T@A
		self.B = c*self.off(self.ATA)
		self.r = r
	def off(self, X):
		#Xcopy = X.copy()
		#np.fill_diagonal(Xcopy, 0.)
		return X - np.diag(np.diag(X))
	def mat_inner(self, A, B):
		return np.trace(B.T@A)
	def f(self, X): #f = ||F(X)-B||_F^2
		return np.linalg.norm((self.F(X)-self.B), ord = 'fro')**2
	def F(self, X):
		return self.off(X) - self.c*self.off(self.A.T@self.off(X)@self.A)
	def F_conj(self, X):
		return self.off(X) - self.c*self.off(self.A@self.off(X)@self.A.T)
	def gradf(self, U): #grad(f)
		return 4.*self.F_conj(self.F(U@U.T)-self.B)@U
	def dgradf(self, U, dX): #differential of grad(f) at U from arg differential dX ; d(grad(f)(U)[dX] = (grad(f))'(U)[dX] = J(U)[dX]
		return 4.*( self.F_conj(self.F(dX@U.T+U@(dX).T))@U + self.F_conj(self.F(U@U.T)-self.B)@dX )
	def dgradf_vectorized(self, U, dX): #(vectorized for scipy iterative solvers) differential of grad(f) 
		return (4.*( self.F_conj(self.F(dX.reshape((self.n,self.r), order = 'F')@U.T+U@(dX.reshape((self.n,self.r), order = 'F')).T))@U + self.F_conj(self.F(U@U.T)-self.B)@dX.reshape((self.n,self.r), order = 'F') ) ).reshape((self.n*self.r,1), order = 'F')

class simrank_ops_another(simrank_ops): #with operators to solve with S = U@U.T
	def __init__(self, A, c, r):
		super().__init__(A, c, r)
		self.B = np.eye(self.n)
	def f(self, X): #f = ||F(X)-B||_F^2
		return np.linalg.norm((self.F(X)-self.B), ord = 'fro')**2
	def F(self, X):
		return X - self.c*self.off(self.A.T@X@self.A)
	def F_conj(self, X):
		return X - self.c*self.A@self.off(X)@self.A.T

class GMRES_scipy:
	def __init__(self, matvec, b, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		self.LinOp = scsp.linalg.LinearOperator((b.shape[0],b.shape[0]), matvec = matvec)
		self.b = b
		self.x_0 = x_0
		self.m = m_Krylov
		self.maxiter = maxiter
		self.eps = eps
		self.printout = printout
		self.cb = lambda res : print(f"GMRES relative residual = {res}")  if printout else None
	def __call__(self):
		u, _ = scsp.linalg.gmres(self.LinOp, self.b, x0=self.x_0, atol=self.eps, restart=self.m, maxiter=self.maxiter, M=None, callback=self.cb, callback_type='pr_norm') 
		return u

class GMRES_custom:
	def __init__(self, matvec, b, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		self.matvec = matvec
		self.b = b
		self.x_0 = x_0
		self.m = m_Krylov
		self.maxiter = maxiter
		self.eps = eps
		self.printout = printout
		self.cb = lambda res : print(f"GMRES relative residual = {res}")
	def __call__(self):
		u, _ = slv.GMRES_m(LinOp=self.matvec, m_Krylov=self.m, x_0=self.x_0, b=self.b, k_max=self.maxiter, eps=self.eps, printout=self.printout)
		return u

#---Newton solver---

def inverse_matvec(LinOp, x, m_Krylov, restarts, solver, eps=1e-10, printout=True):
	if printout : print(f"Starting inverse matvec with : m_Krylov = {m_Krylov} ; restarts = {restarts}")
	st = time.time()
	n, r = x.shape
	solver = solver(LinOp, b=x.reshape((n*r,1), order = 'F'),  x_0=np.zeros((n*r,1)), m_Krylov=m_Krylov, maxiter=restarts, eps=eps, printout=printout)
	u = solver()
	if printout : print(f"Finished inverse matvec for {time.time()-st} s.")
	return u

def Newton(A, c, r, maxiter, gmres_restarts, m_Krylov, solver, stagstop = 1e-5, printout = True):
	iterdata = iterations_data()
	n, _ = A.shape
	np.random.seed(42)
	U = np.random.randn(n,r)
	sops = simrank_ops(A, c, r)
	f_val_prev = sops.f(U@U.T)
	st = time.time()
	for k in range(maxiter):
		dgradf_vec = lambda X : sops.dgradf_vectorized(U, X)
		U = U - inverse_matvec(dgradf_vec, sops.gradf(U), m_Krylov, gmres_restarts, solver, printout=False).reshape((n,r), order = 'F')
		f_val = sops.f(U@U.T)
		iterdata(f_val)
		if (np.abs(f_val - f_val_prev)) < stagstop:
			print(f"Stopped by f(U) stagnation (| f - f_prev | < {stagstop})")
			break
		f_val_prev = f_val
	elapsed = time.time()-st
	solutiondata = [iterdata.iterations, iterdata.funcvals, elapsed]
	return np.eye(n) + sops.off(U@U.T), solutiondata
#---

#--- Gradient method with step splitting---

def getalpha(f, gradf, inner, x, tau, maxiter=100, printout = True): #step splitting
	if printout : print(f"Starting step splitting...")
	alpha = 1.
	kfin = 0
	relax = lambda alpha : f( (x + alpha*gradf(x))@(x + alpha*gradf(x)).T) <= ( f(x@x.T) - tau*alpha*inner(gradf(x), gradf(x)) )
	for k in range(maxiter):
		#print(f"alpha = {alpha}")
		if relax(alpha):
			kfin = k
			break
		alpha = alpha*0.5
	if printout : print(f"Finished with alpha = {alpha} at iter {kfin}")
	return alpha

def Gradmethod(A, c, r, maxiter, printout = True):
	iterdata = iterations_data()
	n, _ = A.shape
	np.random.seed(42)
	U = np.random.randn(n,r)
	sops = simrank_ops(A,c,r)
	st = time.time()
	for k in range(maxiter):
		iterdata(sops.f(U@U.T))
		alpha = getalpha(sops.f, sops.gradf, sops.mat_inner, U, tau=0.5)
		U = U - alpha*sops.gradf(U)
	elapsed = time.time()-st
	solutiondata = [iterdata.iterations, iterdata.funcvals, elapsed]
	return np.eye(n) + sops.off(U@U.T), solutiondata 
#---
