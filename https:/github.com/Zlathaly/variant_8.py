import scipy
import scipy.linalg
import numpy
import matplotlib.pyplot as plt
import time
# %matplotlib inline

'''
Function : TimeEstimator
Purpose  : returns the result of a "functor" with the given "args",
           and the running time of this function
'''
def TimeEstimator(functor, *args):
	launch_time = time.time()
	result = functor(*args)
	stop = time.time()
	return result, stop - launch_time

'''
Class   : MatrixGenerator
Purpose : Contains one public method Generate_Matrix
          that generates a square matrix of a given size from random numbers
		  and multiplies the diagonal elements by a coef
'''
class MatrixGenerator:
	def __generate_random_matrix(size_of_matrix):
		return numpy.random.normal(size=(size_of_matrix, size_of_matrix))
	def __multiply_diagonal(matrix, coef):
		row_len = matrix.shape[1]
		column_len = matrix.shape[0]
		for i in range(column_len):
			if(i == row_len):
				break
			matrix[i][i] *= coef
		return matrix
	def Generate_Matrix(size_of_matrix, coef):
		L_matrix = MatrixGenerator.__generate_random_matrix(size_of_matrix)
		L_matrix = numpy.tril(L_matrix) # lower triangular matrix
		L_matrix = MatrixGenerator.__multiply_diagonal(L_matrix, coef)
		U_matrix = MatrixGenerator.__generate_random_matrix(size_of_matrix)
		U_matrix = numpy.triu(U_matrix) # upper triangular matrix
		U_matrix = MatrixGenerator.__multiply_diagonal(U_matrix, coef)
		A_matrix = numpy.matmul(L_matrix, U_matrix)
		return A_matrix
size = [] # array of matrix sizes
cond = [] # array of numbers is conditional
cond_time_array = [] # array of condition computation time
Solve_time_array = [] # array of computation time of the SLAE solution
norm_r_array = [] # array of residual instrument norms
relative_error_array = [] # array of relative solution errors

for size_of_matrix in range(1, 32):
	A_matrix = MatrixGenerator.Generate_Matrix(size_of_matrix, 0.2) # random illconditioned matrix
	A_cond, cond_time = TimeEstimator(numpy.linalg.cond, A_matrix) # condition number and time of its calculation
	if (A_cond > 1e16):
		continue
	x_generated = numpy.random.rand(size_of_matrix, 1) # a randomly given solution for the matrix A_matrix
	b = numpy.matmul(A_matrix, x_generated) 
	x_calculated, solve_time = TimeEstimator(scipy.linalg.solve, A_matrix, b)
	r = numpy.matmul(A_matrix, x_calculated) - b 
	norm_r = numpy.linalg.norm(r) 
	relative_error = numpy.linalg.norm(x_calculated) / numpy.linalg.norm(x_generated) # relative error of the computed solution
	size.append(size_of_matrix)
	cond.append(A_cond)
	cond_time_array.append(cond_time)
	solve_time_array.append(solve_time)
	norm_r_array.append(norm_r)
	relative_error_array.append(relative_error)
  plt.plot(size, relative_error_array)
plt.show()
