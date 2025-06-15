import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
from memory_profiler import profile

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrixRnd():
	n=100
	A = np.random.randint(low=0, high=10, size=(n,n)) #demo matrix
	print("Random matrix:")
	print(A)
	return A

def GetOnes():
	n=100
	A = np.ones((n,n))
	print("Random matrix:")
	print(A)
	return A

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def ObtainMatrix():
	A = norm1_ColumnNormalize(GetMatrixRnd())
	return A
