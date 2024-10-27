import numpy.linalg
import numpy as np
import time
from scipy.linalg import sqrtm
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent
from sklearn.decomposition import PCA
import autograd.numpy as autonp
from funcs import sum_, diag_
from contrastive import CPCA
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'cyan', 'purple', 'brown', 'pink', 'grey']


def create_cost_and_derivates(manifold, input_data, backend, alpha, mu, bg_data, unexplored_data, method):
    """
    Defines cost and derivatives for the optimization on the selected manifold

    :param manifold: the manifold(Stiefel is default) on which the optimization takes place
    :param input_data: original data samples with n x d dimensions
    :param backend: "autograd" or "numpy"
    :param alpha: hyperparameter controlling the trade-off of original data and background data variance
    :param mu: scaling parameter to make PCA(cPCA) and kurtosis terms comparable
    :param bg_data: the background data samples with m x d dimensions
    :param unexplored_data: the data samples that remain unexplored
    :param method: "PCA + kPP" if there is no background data, "cPCA + kPP" if there is background data

    """

    euclidean_gradient = euclidean_hessian = None
    if backend == "autograd":  # euclidean gradients are computed using Finite-Differences
        if method == "PCA + kPP":
            @pymanopt.function.autograd(manifold)
            def cost(w):
                s1 = 0
                for i in range(unexplored_data.shape[0]):
                    xi = np.expand_dims(unexplored_data[i, :], 1)
                    s1 += (autonp.trace(autonp.linalg.inv(w.T@unexplored_data.T@unexplored_data@w)@w.T@xi@xi.T@w)) ** 2
                return autonp.linalg.norm(input_data - input_data @ w @ w.T) ** 2 + mu * unexplored_data.shape[0] * s1
        elif method == "cPCA + kPP":
            @pymanopt.function.autograd(manifold)
            def cost(w):
                s1 = 0
                for i in range(unexplored_data.shape[0]):
                    xi = np.expand_dims(unexplored_data[i, :], 1)
                    s1 += (autonp.trace(autonp.linalg.inv(w.T@unexplored_data.T@unexplored_data@w)@w.T@xi@xi.T@w)) ** 2
                return autonp.linalg.norm(input_data - input_data @ w @ w.T) ** 2 - \
                       alpha * autonp.linalg.norm(bg_data - bg_data @ w @ w.T) ** 2 + mu * unexplored_data.shape[0] * s1

    elif backend == "numpy":  # Cost and Euclidean Gradient are explicitely provided

        if method == "PCA + kPP":
            @pymanopt.function.numpy(manifold)
            def cost(w):
                l1 = np.linalg.norm(input_data - input_data @ w @ w.T) ** 2
                l2 = mu * input_data.shape[0] * sum_(diag_((input_data @ w)
                                                           @ np.linalg.inv((input_data @ w).T @ (input_data @ w))
                                                           @ (input_data @ w).T) ** 2)
                return l1 + l2

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(w):
                # w is the d x k projection matrix with respect to which we optimise the objective function
                A = w.T @ input_data.T @ input_data @ w
                Ainv = np.linalg.inv(A)
                scal = sqrtm(Ainv) @ w.T @ input_data.T
                scal = np.sqrt(sum_(scal ** 2))
                Mat = (np.ones((input_data.shape[1], 1)) @ scal) * input_data.T
                Mat2 = (np.ones((2, 1)) @ scal) * (w.T @ input_data.T)
                Mat2 = Mat2 @ Mat2.T
                Mat = Mat @ Mat.T
                gr = Mat @ w @ Ainv - (input_data.T @ input_data) @ w @ Ainv @ Mat2 @ Ainv
                gr = 4 * mu * input_data.shape[0] * gr
                return (
                        -2
                        * (input_data.T @ (input_data - input_data @ w @ w.T)
                           + (input_data - input_data @ w @ w.T).T @ input_data
                           ) @ w + gr)

        elif method == "cPCA + kPP":
            @pymanopt.function.numpy(manifold)
            def cost(w):
                return np.linalg.norm(input_data - input_data @ w @ w.T) ** 2 - \
                       alpha * np.linalg.norm(bg_data - bg_data @ w @ w.T) ** 2 + \
                       mu * unexplored_data.shape[0] * sum_(diag_((unexplored_data @ w)
                                                                  @ np.linalg.inv((unexplored_data @ w).T
                                                                                  @ (unexplored_data @ w))
                                                                  @ (unexplored_data @ w).T) ** 2)

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(w):
                A = w.T @ unexplored_data.T @ unexplored_data @ w
                Ainv = np.linalg.inv(A)
                scal = sqrtm(Ainv) @ w.T @ unexplored_data.T
                scal = np.sqrt(sum_(scal ** 2))
                Mat = (np.ones((input_data.shape[1], 1)) @ scal) * unexplored_data.T
                Mat2 = (np.ones((2, 1)) @ scal) * (w.T @ unexplored_data.T)
                Mat2 = Mat2 @ Mat2.T
                Mat = Mat @ Mat.T
                gr = Mat @ w @ Ainv - (unexplored_data.T@unexplored_data)@w@Ainv@Mat2@Ainv
                kurt_grad = 4 * mu * unexplored_data.shape[0] * gr
                return (
                        -2
                        * (input_data.T @ (input_data - input_data @ w @ w.T)
                           + (input_data - input_data @ w @ w.T).T @ input_data
                           ) @ w + 2 * alpha * (bg_data.T @ (bg_data - bg_data @ w @ w.T)
                                                + (bg_data - bg_data @ w @ w.T).T @ bg_data
                                                ) @ w + kurt_grad)

    return cost, euclidean_gradient, euclidean_hessian


class imapce:

    def __init__(self,
                 alpha,
                 mu,
                 ):
        self.alpha = alpha
        self.mu = mu

    def CalculateProjection(self, root, input_data, background_data, method, unexplored_data, iteration, seed):
        """

        :param root: path to save the results
        :param method: "PCA + kPP" if there is no background data, "cPCA + kPP" if there is background data
        :param input_data: original data samples with n x d dimensions
        :param background_data: the background data samples with m x d dimensions
        :param unexplored_data: the data samples that remain unexplored
        :param alpha: hyperparameter controlling the trade-off of original data and background data variance
        :param mu: scaling parameter to make PCA (cPCA) and kurtosis terms comparable
        :param iteration: iteration of the exploration process
        :param seed: seed selected for reproducibility

        """

        backend = "numpy"
        np.random.seed(seed)
        d = input_data.shape[1]
        k = 2  # dimension of the low-dimensional representation
        input_data -= input_data.mean(axis=0)
        manifold = Stiefel(d, k)  # define the manifold for optimization
        t0 = time.time()
        cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(manifold, input_data,
                                                                                backend=backend,
                                                                                alpha=self.alpha, mu=self.mu,
                                                                                bg_data=background_data,
                                                                                unexplored_data=unexplored_data,
                                                                                method=method)
        problem = pymanopt.Problem(
            manifold,
            cost,
            euclidean_gradient=euclidean_gradient,
            euclidean_hessian=euclidean_hessian,
        )  # Initialize the setup on PyManopt
        x0 = np.random.normal(size=(input_data.shape[1], k))  # Set initial point for the optimization
        optimizer = SteepestDescent(root=root, method=method, input_data=input_data,
                                    bg=background_data, alpha=self.alpha, mu=self.mu,
                                    unexplored_data=unexplored_data, explore_iteration=iteration,
                                    verbosity=2)  # Initialize the Steepest Descent Optimizer of PyManopt
        estimated_proj_matrix = optimizer.run(problem, initial_point=x0).point  # Optimize and Compute V
        t1 = time.time() - t0
        print("Calculating the projection took " + str(t1) + " seconds")
        embeddings = input_data@estimated_proj_matrix  # compute data projection along V
        return embeddings


class cPCA_model:

    def __init__(self,
                 num_alphas,
                 max_log_alpha: int = 3,
                 ):
        self.num_alphas = num_alphas
        self.max_log_alpha = max_log_alpha

    def CalculateProjection(self, input_data, background_data, data_labels):
        mdl = CPCA()
        proj = mdl.fit_transform(input_data, background_data, n_alphas=self.num_alphas,
                                 max_log_alpha=self.max_log_alpha, n_alphas_to_return=1, plot=False,
                                 active_labels=data_labels)
        return proj


def CalculateProjection(model_name, input_data, prior_data, **kwargs):
    if model_name == 'IMAPCE':
        model = imapce(kwargs.get('alpha'), kwargs.get('mu'))
        return model.CalculateProjection(root=kwargs.get('root'), method=kwargs.get('method'),
                                         input_data=input_data,
                                         background_data=prior_data,
                                         unexplored_data=kwargs.get('unexplored_data'),
                                         iteration=kwargs.get('iteration'), seed=kwargs.get('seed'))
    elif model_name == 'cPCA':
        if prior_data is None:
            pca = PCA(n_components=2)  # Reduce the data to 2 components for visualization
            proj = pca.fit_transform(input_data)
        else:
            model = cPCA_model(kwargs.get('num_alphas'), kwargs.get('max_log_alpha'))
            proj = np.squeeze(model.CalculateProjection(input_data=input_data, background_data=prior_data,
                                                        data_labels=kwargs.get('data_labels')))
            proj = np.real(proj)
        return proj
