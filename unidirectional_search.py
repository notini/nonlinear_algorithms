import math
from global_definitions import *
from line_search_utils import *

def exact_search(lambdified_fn, lambdified_grad, x_value, direction, alpha = 1, c1 = 0.01, c2 = 0.02):
  lhs = find_left_bracket_interval(lambdified_fn, x_value, direction, alpha)
  rhs = find_right_bracket_value(lambdified_fn, x_value, direction, lhs)
  alpha = golden_section(lambdified_fn, x_value, direction, lhs, rhs)
  return alpha

def armijo_search(lambdified_fn, lambdified_grad, x_value, direction, alpha = 1, c1 = 0.01, c2 = 0.02):
  k = 0
  while True:
    if has_sufficient_decrease(lambdified_fn, lambdified_grad, x_value, alpha, direction, c1):
    #if step_conditions_satisfied(lambdified_fn, lambdified_grad, x_value, alpha, direction, c1, c2):
      break
    alpha = alpha * 0.9
    k += 1
  return alpha

def find_left_bracket_interval(lambdified_fn, x_value, direction, alpha = 1):
  initial_obj_value = lambdified_fn(*x_value)
  k = 1
  while True:
    new_x = x_value + (alpha * direction)
    obj_value = lambdified_fn(*new_x)
    if obj_value < initial_obj_value:
      #print(f"obj value {obj_value} || initial_obj_value {initial_obj_value}")
      lhs = alpha
      break
    alpha = alpha * 0.9
    k += 1
  return lhs

def find_right_bracket_value(lambdified_fn, x_value, direction, lhs):
  k, rhs = 1, float('inf')
  alpha = lhs + 0.1 # set initial alpha as lhs + 1, which is the first interval we check
  lhs_point = x_value + (lhs * direction)
  lhs_obj_value = lambdified_fn(*lhs_point)  
  #print(f'Left bracket obj value is {lhs_obj_value}')
  obj_values = [lhs_obj_value]
  #print(f'Function started to go down at alpha {lhs} with obj value {lhs_obj_value}')

  while True:
  #while k < 5:
    new_x = x_value + (alpha * direction)
    obj_value = lambdified_fn(*new_x)
    if abs(obj_value) - abs(obj_values[-1]) <= epsilon:
      rhs = alpha
      break
    alpha = alpha * 0.9
    k += 1
    obj_values.append(obj_value)
  return rhs

def golden_section(lambdified_fn, x_value, direction, lhs, rhs):
  #print(f'Initial bracket interval is [{lhs}, {rhs}]')
  rho = (3 - math.sqrt(5)) / 2
  iterations = 1
  alpha = None
  while True:
    a1 = lhs + (rho * (rhs - lhs))
    b1 = lhs + (1 - rho) * (rhs - lhs)
    x_value_at_a1 = x_value + (a1 * direction)
    x_value_at_b1 = x_value + (b1 * direction)
    fn_at_a1 = lambdified_fn(*x_value_at_a1)
    fn_at_b1 = lambdified_fn(*x_value_at_b1)
    if abs(fn_at_a1 - fn_at_b1) < epsilon:
      if alpha is None:
        alpha = lhs
      break
    if fn_at_a1 < fn_at_b1:
      rhs = b1
      alpha = lhs
    else:
      lhs = a1
      alpha = rhs
    iterations += 1
  #print(f'Optimal alpha: {alpha} found in {iterations} iterations')
  return alpha
  
