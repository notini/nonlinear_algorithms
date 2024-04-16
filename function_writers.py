def write_fn1_quadractic(x, n):
	fn = 0
	for i in range(1, n):
		fn += (100 * (x[i] - x[i-1]**2) + (1 - x[i-1])**2)
	return fn

def write_fn1(x, n):
	fn = 0
	for i in range(1, n):
		fn += (100 * (x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2)
	return fn

def write_fn2(x, n):
	fn = 0
	for i in range(n):
		fn = fn + x[i]**4 - (16 * x[i]**2) + (5 * x[i])
	return fn

def write_fn3(x, n = None):
	fn = (x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 - 7)**2
	return fn