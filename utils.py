import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify

def compute_gradient(fn, x):
  first_order_derivatives_simpy = np.array([sympy.diff(fn, x_i) for x_i in x.col(0)])  
  partial_deriv = [lambdify(x, first_order_derivatives_simpy[i]) for i in range(len(x))] 
  return partial_deriv

def evaluate_gradient(grad, x):
  evaluated_grad = [grad_fn(*x) for grad_fn in grad]
  return evaluated_grad

def compute_hessian(fn, x):
  h = sympy.hessian(fn, x)
  return h

def evaluate_hessian(lambdified_hessian, x_values):
  evaluated_hessian = lambdified_hessian(*x_values)
  return evaluated_hessian

def write_variable_string(n):
  s = ' '.join([f'x{i}' for i in range(n)])
  return s

def is_downward_direction(H, evaluated_grad):
  v = -np.dot(evaluated_grad, np.dot(evaluated_grad, H)) 
  return v < 0

def fix_positive_definite_matrix(H, epsilon=10**-3):
  if is_positive_definite(H):
    return H
  min_eig = min(np.linalg.eigvals(H))
  new_H = H + (1 + epsilon) * -min_eig * np.identity(H.shape[0])
  return new_H

def is_positive_definite(H):
  try:
    np.linalg.cholesky(H)
    return 1 
  except np.linalg.linalg.LinAlgError as err:
    return 0
  
def pretty_output(x_values, obj_values, alphas):
  K = min([len(x_values), len(obj_values), len(alphas)])
  print(f'Iteration \t\tx \t\tObjective \t\tAlpha')
  for k in range(K):
    print(f'{k} \t{x_values[k]} \t{obj_values[k]} \t{alphas[k]}')