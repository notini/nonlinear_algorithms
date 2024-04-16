import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from utils import *
from function_writers import *
from unidirectional_search import *
from global_definitions import *
from plotting import *

def conjugate_gradient(lambdified_fn, lambdified_grad, initial_x, line_search_fn, max_iter = 5000, initial_alpha = 1):
	x_values, obj_values, alphas = [initial_x], [lambdified_fn(*initial_x)], [None]
	curr_iter = 0
	evaluated_grads = [np.array(evaluate_gradient(lambdified_grad, x_values[0]))]
	directions = [-np.array(evaluated_grads[0])]	
	while curr_iter < max_iter:

		obj_v = lambdified_fn(*x_values[curr_iter])
		print(f'---------------- ITERATION {curr_iter} Obj: {obj_v} x: {x_values[curr_iter]}----------------')
		alpha = line_search_fn(lambdified_fn, lambdified_grad, x_values[curr_iter], directions[curr_iter], initial_alpha)
		alphas.append(alpha)
		new_x = x_values[curr_iter] + (alpha * directions[curr_iter])
		x_values.append(new_x)
		obj_values.append(obj_v)
		old_gradient = evaluated_grads[curr_iter]
		new_gradient = np.array(evaluate_gradient(lambdified_grad, new_x))
		evaluated_grads.append(new_gradient)

		grad_sum = sum(abs(_) for _ in new_gradient)
		if grad_sum < epsilon:
			print('Final solution', x_values[curr_iter], 'objective', lambdified_fn(*x_values[curr_iter]))
			break

		beta = np.linalg.norm(new_gradient) / np.linalg.norm(old_gradient)
		directions.append(-new_gradient + (beta * directions[curr_iter]))

		curr_iter += 1
	return x_values, obj_values, alphas

