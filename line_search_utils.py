import numpy as np
from utils import *

# c1 \in (0, 1)
# c2 should be \in (c1, 1)
def step_conditions_satisfied(lambdified_fn, lambdified_grad, x_value, alpha, direction, c1 = 0.01, c2 = 0.02):
  decrease = has_sufficient_decrease(lambdified_fn, lambdified_grad, x_value, alpha, direction, c1)
  curvature = validate_curvature_condition(lambdified_grad, x_value, alpha, direction, c2)
  return decrease and curvature and c2 > c1

def has_sufficient_decrease(lambdified_fn, lambdified_grad, x_value, alpha, direction, c1 = 0.01):
  candidate_x = x_value + (alpha * direction)
  candidate_fx = lambdified_fn(*candidate_x)
  fx = lambdified_fn(*x_value)
  evaluated_grad = evaluate_gradient(lambdified_grad, x_value)
  grad_direction_dot_prod = np.dot(evaluated_grad, direction)
  #print(f'{candidate_fx} <= {fx + (c1 * alpha * grad_direction_dot_prod)}')
  return candidate_fx <= fx + (c1 * alpha * grad_direction_dot_prod)

def validate_curvature_condition(lambdified_grad, x_value, alpha, direction, c2 = 0.01):
  candidate_x = x_value + (alpha * direction)
  evaluated_grad_at_x = evaluate_gradient(lambdified_grad, x_value)
  evaluated_grad_at_candidate_x = evaluate_gradient(lambdified_grad, candidate_x)
  lhs = np.dot(evaluated_grad_at_candidate_x, direction)
  rhs = c2 * np.dot(evaluated_grad_at_x, direction)
  return lhs >= rhs