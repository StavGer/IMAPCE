"""
Replace the steepest_descent.py file of pymanopt package with this one to have access to the plots of PCA(cPCA),
kurtosis separately as well as their sum and the derivative with respect to V.
"""


import time
from copy import deepcopy
from scipy.linalg.lapack import dgesv
import numpy as np
import matplotlib.pyplot as plt
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer
import random

plt.rcParams['figure.dpi'] = 500

def sum_(X, axis = 0):

    s = np.sum(X, axis=axis, keepdims=True)
    if max(s.shape) == 1:
        return float(s)
    else:
        return s


def diag_(X):
    # diag returns (n,) array, this converts to (n,1)
    return np.expand_dims(np.diag(X), axis = 0).T

class SteepestDescent(Optimizer):
    """Riemannian steepest descent algorithm.

    Perform optimization using gradient descent with line search.
    This method first computes the gradient of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).

    Args:
        line_searcher: The line search method.
    """

    def __init__(self, root, method , input_data, bg = None, alpha = None, mu = None, unexplored_data = None, explore_iteration = None,  line_searcher=None, seed = None, *args, **kwargs):
        """

        :param root : path to save the Loss figures
        :param method : "PCA + kPP" if there is no prior knowledge, "cPCA + kPP" if there is background knowledge associated with background data
        :param input_data : input data of selected dataset
        :param bg : None if there is no background data, else the background data associated with the background knowledge
        :param alpha : contrast trade-off hyperparameter between having high original data variance and low background variance
        :param mu : scaling hyperparameter that accounts for the different scales of PCA(cPCA) and kurtosis terms
        :param unexplored_data : data samples that remain unexplored whose kurtosis is minimized
        :param explore_iteration : iteration of the exploration
        :param line_searcher : defines the line search method
        :param seed : seed for reproducibility
        """
        super().__init__(*args, **kwargs)

        if line_searcher is None:
            self._line_searcher = BackTrackingLineSearcher()
        else:
            self._line_searcher = line_searcher
        self.line_searcher = None
        self._root = root
        self._method = method
        self._input_data = input_data
        self._bg = bg
        self._alpha = alpha
        self._mu = mu
        self._unexplored_data = unexplored_data
        self._iteration = explore_iteration
        self._seed = seed
    # Function to solve optimisation problem using steepest descent.
    def run(
        self, problem, *, initial_point=None, reuse_line_searcher=False
    ) -> OptimizerResult:
        """Run steepest descent algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuse_line_searcher: Whether to reuse the previous line searcher.
                Allows to use information from a previous call to
                :meth:`solve`.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient

        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        line_searcher = self.line_searcher

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            np.random.seed(self._seed)
            x = manifold.random_point()
        else:
            x = initial_point

        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        self._initialize_log(
            optimizer_parameters={"line_searcher": line_searcher}
        )

        # Initialize iteration counter and timer
        iteration = 0
        start_time = time.time()
        total_cost_list = []
        grad_norm_list = []
        l1_cost = []
        l2_cost = []
        convlimit = 1e-6
        while True:
            iteration += 1
            # Calculate new cost, grad and gradient_norm
            total_cost = objective(x)
            if self._method == "PCA + kPP":
                l1 = np.linalg.norm(self._input_data - self._input_data @ x @ x.T) ** 2
                l2 = self._input_data.shape[0] * np.sum((np.atleast_2d(np.diag((self._input_data @ x) @ np.linalg.inv((self._input_data @ x).T @ (self._input_data @ x)) @ (self._input_data @ x).T).T) ** 2))
                l1_cost.append(l1)
                l2_cost.append(l2)
            elif self._method == "cPCA + kPP":
                l1 = np.linalg.norm(self._input_data - self._input_data @ x @ x.T) ** 2 - self._alpha*np.linalg.norm(self._bg - self._bg @ x @ x.T) ** 2
                l2 = self._unexplored_data.shape[0] * np.sum((np.atleast_2d(np.diag((self._unexplored_data @ x) @ np.linalg.inv((self._unexplored_data @ x).T @ (self._unexplored_data @ x)) @ (self._unexplored_data @ x).T).T) ** 2))
                l1_cost.append(l1)
                l2_cost.append(l2)
            grad = gradient(x)
            gradient_norm = manifold.norm(x, grad)
            total_cost_list.append(total_cost)
            grad_norm_list.append(gradient_norm)
            print([str(iteration), str(total_cost), str(gradient_norm)])

            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=total_cost,
                gradient_norm=gradient_norm,
            )

            # Descent direction is minus the gradient
            desc_dir = -grad
            oldV = x
            # Perform line-search
            step_size, x = line_searcher.search(
                objective, manifold, x, desc_dir, total_cost, -(gradient_norm**2)
            )
            V = x
            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                step_size=step_size,
                gradient_norm=gradient_norm,
                iteration=iteration,
            )
            if stopping_criterion or (sum_((oldV - V) ** 2) < convlimit).all() :
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                    cost_arr = np.asarray(total_cost_list)
                    norm_grad_arr = np.asarray(grad_norm_list)
                    cost_iter = np.arange(cost_arr.shape[0])
                    if self._method == "PCA + kPP" or self._method == "cPCA + kPP":
                        l1_arr = np.asarray(l1_cost)
                        l2_arr = np.asarray(l2_cost)
                        plt.subplot(4,1,1)
                        plt.plot(cost_iter, cost_arr)
                        yval = np.array([np.min(cost_arr), np.max(cost_arr)])
                        plt.yticks(yval)
                        plt.xlabel("Iteration")
                        plt.ylabel("Cost")
                        plt.yscale('log')
                        plt.subplot(4,1,2)
                        plt.plot(cost_iter,norm_grad_arr)
                        yval = np.array([np.min(norm_grad_arr), np.max(norm_grad_arr)])
                        plt.yticks(yval)
                        plt.xlabel("Iteration")
                        plt.ylabel("Grad")
                        plt.yscale('log')
                        plt.subplot(4,1,3)
                        plt.plot(cost_iter,l1_arr)
                        yval = np.array([np.min(l1_arr), np.max(l1_arr)])
                        plt.yticks(yval)
                        plt.xlabel("Iteration")
                        plt.ylabel("L1")
                        plt.yscale('log')
                        plt.subplot(4,1,4)
                        plt.plot(cost_iter,l2_arr)
                        yval = np.array([np.min(l2_arr), np.max(l2_arr)])
                        plt.yticks(yval)
                        plt.xlabel("Iteration")
                        plt.ylabel("Kurtosis")
                        plt.tight_layout()
                    plt.savefig(self._root + 'Losses' + str(self._iteration))
                break
            oldV = V

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=objective(x),
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )
