# BLESS: Bottom-up leverage score sampling
Python code implementing the ridge leverage score sampling algorithm __BLESS__ presented in:
[On Fast Leverage Score Sampling and Optimal Learning](https://arxiv.org/abs/1810.13258) (NIPS 2018).
The implementation can exploit both GPU and CPU resources.

## Modules required

* `numpy` is a mandatory requirement
* `cupy` is optionally required to use GPU resources
* `tqdm` is optionally required for better progress bars

## Installation
The algorithm BLESS is contained in a single python file, and can be installed simply by placing it in your `PYTHONPATH` (e.g. your current folder)

## Usage
The algorithm is implemented by the function
```python
bless(X, similarity_func, lam_final=2.0, qbar=2, random_state=None, H=None, force_cpu=False, verbose=True)
```
It returns an *(eps, lambda)*-accurate dictionary of Nystrom centers sampled according to approximate RLS.

Given data `X`, a similarity function, and its related similarity matrix `similarity_function(X, X)`,
an *(eps, lambda)*-accurate dictionary approximates all principal components of the similarity matrix
with a singular value larger than lambda, up to a *(1+eps)* multiplicative error.

The algorithm is introduced and analyzed in [Rudi et al. 18], for a more formal
definition of *(eps, lambda)*-accuracy and other potential uses see [Calandriello et al. 18].


**Input:**

* `X` : Input data, as an *ndarray*-like (*n* x *m*) object.

* `similarity_func`: callable similarity (kernel) function between points. Denoting with `K` the kernel, it must satisfy the interface\
`similarity_func(X_1) = similarity_func(X_1, X_1)`,\
`similarity_func(X_1, X_2) = K(X_1, X_2)`,\
`similarity_func.diag(X_1) = diag(K(X_1, X_1))`,\
This interface is inspired by *scikit-learn*'s implementation of kernel functions in Gaussian Processes.
Any of the kernels provided by *sklearn* (e.g. `sklearn.gaussian_process.kernels.RBF` or
`sklearn.gaussian_process.kernels.PairwiseKernel`) should work out of the box.

* `lam_final`: final lambda (i.e. as in (*eps*, *lambda*)-accuracy) desired. Roughly, the final dictionary will approximate all eigenvalues larger than lam\_final, and therefore smaller lam_final creates larger, more accurate dictionaries.

* `qbar`: Oversampling parameter. BLESS includes centers in the output dictionary using approximate RLS sampling.
        The *qbar* >= 1 parameter is used to increase the sampling probabilities and sample size by a qbar factor.
        This reduces variance and the negative effects of randomness on the accuracy of the algorithm,
        but at the same time the size of the output dictionary scales linearly with qbar.
        Empirically, a small factor *qbar* = [2,10] seems to work. It is suggested to start with a small number and
        increase if the algorithm fails to terminate or is not accurate.
        For more details, see [Rudi et al. 2018](https://arxiv.org/abs/1810.13258)

* `random_state`: Random number generator (RNG) used for the algorithm. By default, if random_state is not provided, a *numpy*'s *RandomState* with default seeding is used. If a *numpy*'s *RandomState* is passed, it is used as RNG. If an int is passed, it is used to seed a *RandomState*.

* `H`: Number of iterations of the algorithm (i.e. rounds of reduction from *n* to *lam_final*), defaults to *log(n)* if `H = None`.

* `force_cpu`: If True, forces the use of CPU. In this case, BLESS does not even attempt to load *cupy* as a GPU driver, and can be used without *cupy* installed.

* `verbose`: Controls verbosity of debug output, including progress bars.
   The progress bar reports:
   - `lam`: lambda value of the current iteration
   - `m`: current size of the dictionary (number of centers contained)
   - `m_expected`: expected size of the dictionary before sampling
   - `probs_dist`:(mean, max, min) of the approximate RLSs at the current iteration

**Output:**

An *(eps, lambda)-accurate dictionary* dictionary `centers_dict` (with high probability).
If `centers_dict` contains *m* entries then the outpout fields are as follows

* `centers_dict.idx`: the indices of the *m* selected samples in the input dataset `X`
* `centers_dict.X`': the (*m* x *d*) *numpy.ndarray* containing the selected samples
* `centers_dict.probs`: the probabilities (i.e. approximate RLSs) used to sample the dictionary
* `lam`: the final lambda accuracy
* `qbar`: the `qbar` used to sample the dictionary, as a proxy for the *eps*-accuracy

**Using the output:**

A few functions that uses the output of BLESS to build some objects useful in different learning applications are also implemented.\
In particular the function
```python
get_nystrom_embeddings(X, centers_dict, similarity_func, force_cpu=False)
```
computes the Nystrom embeddings using the centers sampled by BLESS.

The function
```python
get_nystrom_matrix_approx(X, centers_dict, similarity_func, force_cpu=False)
```
computes the Nystrom matrix approximation using the centers sampled by BLESS.

The function
```python
get_nystrom_PCA(X, centers_dict, similarity_func, k=-1, force_cpu=False)
```
computes the PCA with the Nystrom approximation using the centers sampled by BLESS.

### Usage Example

The following example is included in the file, so simply run `python bless.py` to test the installation.

```python
    from sklearn.gaussian_process.kernels import RBF
    X_test = np.random.randn(30000, 10)
    r = np.random.RandomState(42)

    D_test = bless(X_test, RBF(length_scale=10), 10, 10, r, 10, force_cpu=True)

    try:
        import cupy
        D_test2 = bless(X_test, RBF(length_scale=10), 10, 10, r, 10, force_cpu=False)
    except ImportError:
        print("cupy not found, defaulting to numpy")
```

### References
[1] Rudi A, Calandriello D, Carratino L, Rosasco L.
           On fast leverage score sampling and optimal learning. In NeurIPS 2018

```
@inproceedings{rudi2018fast,
  title={On fast leverage score sampling and optimal learning},
  author={Rudi, Alessandro and Calandriello, Daniele and Carratino, Luigi and Rosasco, Lorenzo},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5672--5682},
  year={2018}
}
```

[2] Calandriello D, Lazaric A, Valko M.
           Distributed adaptive sampling for kernel matrix approximation. AI&STATS 2017.

```
@inproceedings{calandriello2017distributed,
  title={Distributed adaptive sampling for kernel matrix approximation},
  author={Calandriello, Daniele and Lazaric, Alessandro and Valko, Michal},
  booktitle={Artificial Intelligence and Statistics},
  pages={1421--1429},
  year={2017}
}
```
