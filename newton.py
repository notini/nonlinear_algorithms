import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from utils import *
from function_writers import *
from unidirectional_search import *
from global_definitions import *
from plotting import *

def newton(x, fn, lambdified_fn, lambdified_grad, initial_x, line_search_fn, max_iter = 5000, initial_alpha = 1):
	x_values, obj_values, alphas = [initial_x], [lambdified_fn(*initial_x)], [None]
	curr_iter = 0
	hess = compute_hessian(fn, x)
	inverse_hessian = hess.inv()
	lambdified_inverse_hessian = lambdify(x, inverse_hessian)

	while curr_iter < max_iter:
		print(f'---------------- ITERATION {curr_iter} Obj: {obj_values[curr_iter]} x: {x_values[curr_iter]}----------------')

		evaluated_grad = evaluate_gradient(lambdified_grad, x_values[curr_iter])
		grad_sum = sum(abs(_) for _ in evaluated_grad)
		if grad_sum < epsilon:
			print('Final solution', x_values[curr_iter], 'objective', lambdified_fn(*x_values[curr_iter]))
			break

		inverse_evaluated_hessian = evaluate_hessian(lambdified_inverse_hessian, x_values[curr_iter])
		is_psd = is_positive_definite(inverse_evaluated_hessian)
		if not is_psd:
			inverse_evaluated_hessian = fix_positive_definite_matrix(inverse_evaluated_hessian)
		direction = -np.matmul(inverse_evaluated_hessian, evaluated_grad)

		alpha = line_search_fn(lambdified_fn, lambdified_grad, x_values[curr_iter], direction, initial_alpha)
		alphas.append(alpha)
		# alpha = armijo_search(lambdified_fn, lambdified_grad, x_values[curr_iter], direction)
		new_x = x_values[curr_iter] + (alpha * direction)
		x_values.append(new_x)
		obj_values.append(lambdified_fn(*new_x))
		
		curr_iter += 1
	return x_values, obj_values, alphas

#plot_travel_path_countour(x_values, lambdified_fn)

#pretty_output(x_values, obj_values, alphas)