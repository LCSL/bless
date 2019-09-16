# MIT License
#
# Copyright (c) 2017 Laboratory for Computational and Statistical Learning
#
# authors: Daniele Calandriello, Luigi Carratino
# email:   daniele.calandriello@iit.it
# Website: http://lcsl.mit.edu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from collections import namedtuple

CentersDictionary = namedtuple('CentersDictionary', ('idx', 'X', 'probs', 'lam', 'qbar'))


def __load_gpu_module(force_cpu: bool):
    """Helper function that tries to detect if cupy is present to allow use of GPUs.
    If force_cpu is True, does not even try to load cupy and simply return
    numpy. This way we can run BLESS even if cupy is not installed.
    If False, we first try to load cupy and fall back to numpy if we cannot
    detect it.
    The returned xp must also provide a .asnumpy method that converts
    whatever internal array representation xp uses
    to a numpy ndarray"""

    xp = np
    # this is essentially an identity function for numpy arrays
    xp.asnumpy = np.asarray

    if not force_cpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            print("cupy not found, defaulting to numpy")

    return xp


def __get_progress_bar(total=-1, disable=False):
    """Helper function to get a tqdm progress bar (or a simple fallback otherwise)"""
    class ProgBar(object):
        def __init__(self, total=-1, disable=False):
            self.disable = disable
            self.t = 0
            self.total = total
            self.debug_string = ""

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        def set_postfix(self, **kwargs):
            self.debug_string = ""
            for arg in kwargs:
                self.debug_string += "{}={} ".format(arg, kwargs[arg])

        def update(self):
            if not self.disable:
                self.t += 1
                print_str = "{}".format(self.t)

                if self.total > 0:
                    print_str += "/{}".format(self.total)

                print_str += ": {}".format(self.debug_string)

                if len(print_str) < 80:
                    print_str = print_str + " "*(80 - len(print_str))

                print(print_str, end='\r', flush=True)

            if self.t == self.total:
                print("")

    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total, disable=disable)
    except ImportError:
        progress_bar = ProgBar(total=total, disable=disable)

    return progress_bar


def __stable_invert_root(U: np.ndarray, S: np.ndarray):
    n = U.shape[0]

    assert U.shape == (n, n)
    assert S.shape == (n,)

    # threshold formula taken from pinv2's implementation of numpy/scipy
    thresh = S.max() * max(S.shape) * np.finfo(S.dtype).eps
    stable_eig = np.logical_not(np.isclose(S, 0., atol=thresh))
    m = sum(stable_eig)

    U_thin = U[:, stable_eig]
    S_thin = S[stable_eig]

    assert U_thin.shape == (n, m)
    assert S_thin.shape == (m,)

    S_thin_inv_root = (1 / np.sqrt(S_thin)).reshape(-1, 1)

    return U_thin, S_thin_inv_root


def compute_tau(centers_dict: CentersDictionary,
                X: np.ndarray,
                similarity_func: callable,
                lam_new: float,
                force_cpu=False):
    """Given a previosuly computed (eps, lambda)-accurate dictionary, it computes estimates
    of all RLS using the estimator from Calandriello et al. 2017"""
    xp = __load_gpu_module(force_cpu)

    diag_norm = np.asarray(similarity_func.diag(X))

    # (m x n) kernel matrix between samples in dictionary and dataset X
    K_DU = xp.asarray(similarity_func(centers_dict.X, X))

    # the estimator proposed in Calandriello et al. 2017 is
    # diag(XX' - XX'S(SX'XS + lam*I)^(-1)SXX')/lam
    # here for efficiency we collect an S inside the inverse and compute
    # diag(XX' - XX'(X'X + lam*S^(-2))^(-1)XX')/lam
    # note that in the second term we take care of dropping the rows/columns of X associated
    # with 0 entries in S
    U_DD, S_DD, _ = np.linalg.svd(xp.asnumpy(similarity_func(centers_dict.X, centers_dict.X)
                                             + lam_new * np.diag(centers_dict.probs)))

    U_DD, S_root_inv_DD = __stable_invert_root(U_DD, S_DD)

    E = xp.asarray(S_root_inv_DD * U_DD.T)

    # compute (X'X + lam*S^(-2))^(-1/2)XX'
    X_precond = E.dot(K_DU)

    # the diagonal entries of XX'(X'X + lam*S^(-2))^(-1)XX' are just the squared
    # ell-2 norm of the columns of (X'X + lam*S^(-2))^(-1/2)XX'
    tau = (diag_norm - xp.asnumpy(xp.square(X_precond, out=X_precond).sum(axis=0))) / lam_new

    assert np.all(tau >= 0.), ('Some estimated RLS is negative, this should never happen. '
                               'Min prob: {:.5f}'.format(np.min(tau)))

    return tau


def reduce_lambda(X: np.ndarray,
                  similarity_func: callable,
                  centers_dict: CentersDictionary,
                  lam_new: float,
                  random_state: np.random.RandomState,
                  qbar=None,
                  force_cpu=False):
    """Given a previosuly computed (eps, lambda)-accurate dictionary and a lambda' < lambda parameter,
     it constructs an (eps, lambda')-accurate dictionary using approximate RLS sampling.
    """

    n, d = X.shape

    if qbar is None:
        qbar = centers_dict.qbar

    red_ratio = centers_dict.lam / lam_new

    assert red_ratio >= 1.

    diag = np.asarray(similarity_func.diag(X))

    # compute upper confidence bound on RLS of each sample, overestimate (oversample) by a qbar factor
    # to boost success probability at the expenses of a larger sample (dictionary)
    ucb = np.minimum(qbar * diag / (diag + lam_new), 1.)

    U = np.asarray(random_state.rand(n)) <= ucb
    u = U.sum()

    assert u > 0, ('No point selected during uniform sampling step, try to increase qbar. '
                   'Expected number of points: {:.3f}'.format(n * ucb))

    X_U = X[U, :]

    # taus are RLS
    tau = compute_tau(centers_dict, X_U, similarity_func, lam_new, force_cpu)

    # RLS should always be smaller than 1
    tau = np.minimum(tau, 1.)

    # same as before, oversample by a qbar factor
    probs = np.minimum(qbar * tau, ucb[U]) / ucb[U]

    assert np.all(probs >= 0.), ('Some estimated probability is negative, this should never happen. '
                                 'Min prob: {:.5f}'.format(np.min(probs)))

    deff_estimate = probs.sum()/qbar
    assert qbar*deff_estimate >= 1., ('Estimated deff is smaller than 1, you might want to reconsider your kernel. '
                                      'deff_estimate: {:.3f}'.format(qbar*deff_estimate))

    selected = np.asarray(random_state.rand(u)) <= probs

    s = selected.sum()

    assert s > 0, ('No point selected during RLS sampling step, try to increase qbar. '
                   'Expected number of points (qbar*deff): {:.3f}'.format(np.sum(probs)))

    D_new = CentersDictionary(idx=U.nonzero()[0][selected.nonzero()[0]],
                              X=X_U[selected, :],
                              probs=probs[selected],
                              lam=lam_new,
                              qbar=qbar)

    return D_new


def bless(X, similarity_func, lam_final=2.0, qbar=2, random_state=None, H=None, force_cpu=False, verbose=True):
    """
    Returns a (eps, lambda)-accurate dictionary of Nystrom centers sampled according to approximate RLS.

    Given data X, a similarity function, and its related similarity matrix similarity_function(X, X),
    an (eps, lambda)-accurate dictionary approximates all principal components of the similarity matrix
    with a singular value larger than lambda, up to a (1+eps) multiplicative error.

    The algorithm is introduced and analyzed in [Rudi et al. 18], for a more formal
    definition of (eps, lambda)-accuracy and other potential uses see [Calandriello et al. 18].

    Parameters
    ----------
    X : array_like
        Input data, as an ndarray-like (n x m) object.

    similarity_func: callable
        similarity (kernel) function between points. Denoting with K the kernel, it must satisfy the interface
        similarity_func(X_1) = similarity_func(X_1, X_1)
        similarity_func(X_1, X_2) = K(X_1, X_2)
        similarity_func.diag(X_1) = diag(K(X_1, X_1))
        This interface is inspired by scikit-learn's implementation of kernel functions in Gaussian Processes.
        Any of the kernels provided by sklearn (e.g. sklearn.gaussian_process.kernels.RBF or
        sklearn.gaussian_process.kernels.PairwiseKernel) should work out of the box.

    lam_final: float
        final lambda (i.e. as in (eps, lambda)-accuracy) desired. Roughly, the final dictionary will approximate
        all principal components with a singular value larger than lam_final, and therefore smaller lam_final
        creates larger, more accurate dictionaries.

    qbar: float
        Oversampling parameter used during BLESS's step of random RLS sampling.
        The qbar >= 1 parameter is used to increase the sampling probabilities and sample size by a qbar factor.
        This linearly increases the size of the output dictionary, making the algorithm less memory and time efficient,
        but reduces variance and the negative effects of randomness on the accuracy of the algorithm.
        Empirically, a small factor qbar = [2,10] seems to work. It is suggested to start with a small number and
        increase if the algorithm fails to terminate or is not accurate. 
        For more details, see [Rudi et al. 2018](https://arxiv.org/abs/1810.13258)

    random_state: np.random.RandomState or int or None
        Random number generator (RNG) used for the algorithm. 
        By default, if random_state is not provided, a numpy's RandomState with default seeding is used. 
        If a numpy's RandomState is passed, it is used as RNG. If an int is passed, it is used to seed a RandomState.

    H: int
        Number of iterations (i.e. rounds of reduction from n to lam_final), defaults to log(n) if H=None.

    force_cpu: bool
        If True, forces the use of CPU. In this case, BLESS does not even attempt
        to load cupy as a GPU driver, and can be used without cupy installed.

    verbose: int
        Controls verbosity of debug output, including progress bars.
        The progress bar reports:
        - lam: lambda value of the current iteration
        - m: current size of the dictionary (number of centers contained)
        - m_expected: expected size of the dictionary before sampling
        - probs_dist: (mean, max, min) of the approximate RLSs at the current iteration



    Returns
    -------
    CentersDictionary
        An (eps, lambda)-accurate dictionary centers_dict (with high probability).
        If centers_dict contains m entries then the output fields are as follow

        centers_dict.idx`: the indices of the m selected samples in the input dataset `X`
        centers_dict.X': the (m x d) numpy.ndarray containing the selected samples
        centers_dict.probs: the probabilities (i.e. approximate RLSs) used to sample the dictionary
        lam: the final lambda accuracy
        qbar: the qbar used to sample the dictionary, as a proxy for the `eps`-accuracy

    Raises
    ------
    AssertionError
        If some of the internal checks fail, which usually indicate the high probability event did not happen
        and some parameter should be corrected

    ValueError
        If the supplied RNG is not supported.

    References
    ------
    .. [1] Rudi A, Calandriello D, Carratino L, Rosasco L.
           On fast leverage score sampling and optimal learning. In NeurIPS 2018

    .. [2] Calandriello D, Lazaric A, Valko M.
           Distributed adaptive sampling for kernel matrix approximation. AI&STATS 2017.
    """

    n, d = X.shape

    H = H if H is not None else np.ceil(np.log(n)).astype('int')

    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, (int, np.int)):
        rng = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise ValueError('Cannot understand what you passed as a random number generator.')
    
    diag_norm = np.asarray(similarity_func.diag(X))
    ucb_init = qbar * diag_norm / n

    selected_init = rng.rand(n) <= ucb_init

    # force at least one sample to be selected
    selected_init[0] = 1

    D = CentersDictionary(idx=selected_init.nonzero(),
                   X=X[selected_init, :],
                   probs=np.ones(np.sum(selected_init)) * ucb_init[selected_init],
                   lam=n,
                   qbar=qbar)

    lam_sequence = list(np.geomspace(lam_final, n, H))

    # discard n from the list, we already used it to initialize
    lam_sequence.pop()

    with __get_progress_bar(total=len(lam_sequence), disable=not(verbose)) as t:
        while len(lam_sequence) > 0:
            lam_new = lam_sequence.pop()
            D = reduce_lambda(X, similarity_func, D, lam_new, rng, force_cpu=force_cpu)
            t.set_postfix(lam=int(lam_new),
                          m=len(D.probs),
                          m_expected=int(D.probs.mean()*n),
                          probs_dist=f"({D.probs.mean()/qbar:.4}, {D.probs.max()/qbar:.4}, {D.probs.min()/qbar:.4})")
            t.update()

    return D


def get_nystrom_embeddings(X, centers_dict, similarity_func, force_cpu=False):
    xp = __load_gpu_module(force_cpu)
    K_XD = xp.asarray(similarity_func(X, centers_dict.X))

    U_DD, S_DD, _ = np.linalg.svd(xp.asnumpy(similarity_func(centers_dict.X, centers_dict.X)))
    U_DD, S_root_inv_DD = __stable_invert_root(U_DD, S_DD)

    K_DD_inv_sqrt = xp.asarray(U_DD * S_root_inv_DD.T)

    return K_XD.dot(K_DD_inv_sqrt)


def get_nystrom_matrix_approx(X, centers_dict, similarity_func, force_cpu=False):
    B = get_nystrom_embeddings(X, centers_dict, similarity_func, force_cpu)
    return B.dot(B.T)


def get_nystrom_PCA(X, centers_dict, similarity_func, k=-1, force_cpu=False):
    B = get_nystrom_embeddings(X, centers_dict, similarity_func, force_cpu)
    if k > B.shape[1]:
        raise ValueError('requesting k={} principal components, but the centers dictionary can only'
                         'approximate m={} components.'.format(k, B.shape[1]))
    U, Sigma, _ = np.linalg.svd(B,
                                full_matrices=False,
                                compute_uv=True)
    return np.dot(U, np.diag(Sigma))


if __name__ == "__main__":
    from sklearn.gaussian_process.kernels import RBF
    X_test = np.random.randn(30000, 10)
    r = np.random.RandomState(42)

    D_test = bless(X_test, RBF(length_scale=10), 10, 10, r, 10, force_cpu=True)

    try:
        import cupy
        D_test2 = bless(X_test, RBF(length_scale=10), 10, 10, r, 10, force_cpu=False)
    except ImportError:
        print("cupy not found, defaulting to numpy")

    pass
