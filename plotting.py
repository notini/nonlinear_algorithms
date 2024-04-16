import matplotlib.pyplot as plt 
import numpy as np

def plot_travel_path_countour(x_values, lambdified_fn):
	x = np.linspace(-2, 2, 100)                         
	y = np.linspace(-2, 2, 100)                         
	x, y = np.meshgrid(x, y)                            
	fig, ax = plt.subplots()
	CS = ax.contour(x, y, lambdified_fn(x, y), 200)
	ax.set_title('labels at selected locations')  
	plt.plot(x_values, x_values, 'x')
	plt.show()

