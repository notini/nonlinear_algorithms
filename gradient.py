import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from utils import *
from function_writers import *
from unidirectional_search import *
from global_definitions import *
from plotting import *

def gradient_method(lambdified_fn, lambdified_grad, initial_x, line_search_fn, max_iter = 5000, initial_alpha = 1):
  curr_iter = 0
  x_values, obj_values, alphas = [initial_x], [lambdified_fn(*initial_x)], [None]
  while curr_iter < max_iter:
    obj_v = lambdified_fn(*x_values[curr_iter])
    print(f'---------------- ITERATION {curr_iter} Obj: {obj_v} x: {x_values[curr_iter]}----------------')

    evaluated_grad = evaluate_gradient(lambdified_grad, x_values[curr_iter])
    grad_sum = sum(abs(_) for _ in evaluated_grad)
    if grad_sum < epsilon:
      print('Final solution', x_values[curr_iter], 'objective', lambdified_fn(*x_values[curr_iter]))
      break

    direction = -np.array(evaluated_grad)

    # alpha = exact_search(lambdified_fn, lambdified_grad, x_values[curr_iter], direction)
    alpha = line_search_fn(lambdified_fn, lambdified_grad, x_values[curr_iter], direction, initial_alpha)
    alphas.append(alpha)
    new_x = x_values[curr_iter] + (alpha * direction)
    #print(f'new_x = {x_values[curr_iter]} + {alpha} * {direction}')
    x_values.append(new_x)
    obj_values.append(obj_v)
    
    curr_iter += 1
  return x_values, obj_values, alphas
	