import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from utils import *
from function_writers import *
from unidirectional_search import *
from global_definitions import *

def dfp(lambdified_fn, lambdified_grad, initial_x, line_search_fn, max_iter = 5000, initial_alpha = 1):
	curr_iter = 0
	x_values, obj_values, alphas = [initial_x], [lambdified_fn(*initial_x)], [None]	
	n = len(initial_x)
	H = [[1 if i == j else 0 for i in range(n)] for j in range(n)]
	
	while curr_iter < max_iter:
		obj_v = lambdified_fn(*x_values[curr_iter])
		print(f'---------------- ITERATION {curr_iter} Obj: {obj_v} x: {x_values[curr_iter]}----------------')

		evaluated_grad = evaluate_gradient(lambdified_grad, x_values[curr_iter])
		grad_sum = sum(abs(_) for _ in evaluated_grad)
		if grad_sum < epsilon:
			print('Final solution', x_values[curr_iter], 'objective', lambdified_fn(*x_values[curr_iter]))
			break

		direction = -np.matmul(evaluated_grad, H)

		alpha = line_search_fn(lambdified_fn, lambdified_grad, x_values[curr_iter], direction, initial_alpha)
		# alpha = armijo_search(lambdified_fn, lambdified_grad, x_values[curr_iter], direction)
		alphas.append(alpha)
		new_x = x_values[curr_iter] + (alpha * direction)
		x_values.append(new_x)
		obj_values.append(obj_v)

		new_evaluated_grad = evaluate_gradient(lambdified_grad, new_x)
		g = np.array([[new_evaluated_grad[i] - evaluated_grad[i] for i in range(n)]]).reshape(n, 1)
		p = np.array(alpha * direction).reshape(n, 1)
		ppT = np.dot(p, p.T)
		pTg = np.dot(p.T, g)

		Hg = np.dot(H, g)
		gTH = np.dot(g.T, H)
		HggTH = np.dot(Hg, gTH)
		gTHg = np.dot(gTH, g)

		first_term = ppT / pTg
		second_term = HggTH / gTHg	

		H = H + first_term - second_term
		
		curr_iter += 1
	return x_values, obj_values, alphas