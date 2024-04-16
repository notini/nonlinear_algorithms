# Nonlinear Optimization Algorithms

This repository contains personal implementations of classical global optimization algorithms. Five algorithms are present here: the Gradient, Conjugate-Gradient, Newton, Davidon-Fletcher-Powell, and Broyden-Fletcher-Goldfarb-Shanno Methods. The goal is for this repository to serve as a reference for when you may be working on your own implementation of these algorithms. Not a lot of thought was put into being as efficient as possible since the purpose here was purely educational.

Three functions are originally available, but you may write your own following the example of the provided ones.

Gradient and Hessian calculations are obtained dynamically, using the Sympy package. Therefore, you do not need to provide functions for those yourself.

Each algorithm can be used with exact (Golden Section) and inexact (Armijo with qualification criteria) line search methods. To swap between line search methods, simply change the commented line on the respective cell for your desired algorithm. You may also choose a larger value for n. Examples of this can be seen in the provided Jupyter Notebook.

Any feedback or questions, feel free to reach out at vitornotini@gmail.com.
