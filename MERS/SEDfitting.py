import numpy as np
# Parallel processing with joblib for better performance
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.linalg import orth, null_space



def cg(
    Amat,
    bvec,
    maxiters=1000,
    abs_tol=1e-10,
    use_norm_tol=False,
    x0=None,
    linear_op=None,
    comm=None,
):
    """
    Simple Conjugate Gradient solver that operates in serial. This uses the
    same algorithm as `cg_mpi()` and so can be used for testing/comparison of
    results.

    Note that this function will still permit threading used within numpy.

    Parameters:
        Amat (array_like):
            Linear operator matrix.
        bvec (array_like):
            Right-hand side vector.
        maxiters (int):
            Maximum number of iterations of the solver to perform before
            returning.
        abs_tol (float):
            Absolute tolerance on each element of the residual. Once this
            tolerance has been reached for all entries of the residual vector,
            the solution is considered to have converged.
        use_norm_tol (bool):
            Whether to use the tolerance on each element (as above), or an
            overall tolerance on the norm of the residual.
        x0 (array_like):
            Initial guess for the solution vector. Will be set to zero
            otherwise.
        linear_op (func):
            If specified, this function will be used to operate on vectors,
            instead of the Amat matrix. Must have call signature `func(x)`.
        comm (MPI communicator):
            If specified, the CG solver will be run only on the root worker,
            but the

    Returns:
        x (array_like):
            Solution vector for the full system.
    """
    # MPI worker ID
    myid = 0
    if comm is not None:
        myid = comm.Get_rank()

    # Use Amat as the linear operator if function not specified
    if linear_op is None:
        linear_op = lambda v: Amat @ v

    # Initialise solution vector
    if x0 is None:
        x = np.zeros_like(bvec)
    else:
        assert x0.shape == bvec.shape, "Initial guess x0 has a different shape to bvec"
        assert x0.dtype == bvec.dtype, "Initial guess x0 has a different type to bvec"
        x = x0.copy()

    # Calculate initial residual
    # NOTE: linear_op may have internal MPI calls; we assume that it
    # handles synchronising its input itself, but that only the root
    # worker receives a correct return value.

    r = bvec - linear_op(x)
    pvec = r[:]

    # Blocks indexed by i,j: y = A . x = Sum_j A_ij b_j
    niter = 0
    finished = False
    while niter < maxiters and not finished:

        try:
            if myid == 0:
                # Root worker checks for convergence
                if use_norm_tol:
                    # Check tolerance on norm of r
                    if np.linalg.norm(r) < abs_tol:
                        finished = True
                else:
                    # Check tolerance per array element
                    if np.all(np.abs(r) < abs_tol):
                        finished = True

            # Broadcast finished flag to all workers (need to use a non-immutable type)
            finished_arr = np.array(
                [
                    finished,
                ]
            )
            if comm is not None:
                comm.Bcast(finished_arr, root=0)
                finished = bool(finished_arr[0])
            if finished:
                break

            # Do CG iteration
            r_dot_r = np.dot(r.T, r)
            A_dot_p = linear_op(pvec)  # root worker will broadcast correct pvec

            # Only root worker needs to do these updates; other workers idle
            if myid == 0:
                pAp = pvec.T @ A_dot_p
                alpha = r_dot_r / pAp

                x = x + alpha * pvec
                r = r - alpha * A_dot_p

                # Update pvec
                beta = np.dot(r.T, r) / r_dot_r
                pvec = r + beta * pvec

            # Update pvec on all workers
            if comm is not None:
                comm.Bcast(pvec, root=0)
            
            # Increment iteration
            niter += 1
        except:
            raise

    if comm is not None:
        comm.barrier()

    # Synchronise solution across all workers
    if comm is not None:
        comm.Bcast(x, root=0)
    return x


def linear_fit_1D(data, basis, return_loss=True):
    """
    Perform linear fit for the spectrum of a single pixel.

    Parameters:
    - data: 1D array of data, shape (n_freqs,)
    - basis: 2D array of basis functions, shape (n_freqs, n_basis)
    Returns:
    - coefficients: 1D array of fit coefficients, shape (n_basis,)
    """
    A = basis
    b = data
    ATA = A.T @ A + 1e-8 * np.eye(A.shape[1]) 
    ATb = A.T @ b
    coefficients = cg(ATA, ATb)
    if return_loss:
        residuals = b - A @ coefficients
        # Calculate RMS fractional error 
        assert not np.any(np.isnan(residuals)), "NaN values detected in residuals"
        assert np.all(b > 1e-10), "b is too small to define RMS fractional error.."
        loss = np.sqrt(np.mean( (residuals / np.abs(b) ) ** 2))  # Define the loss as the root of mean squared fractional error
        return coefficients, loss

    return coefficients

def linear_model(basis, coefficients):
    """
    Compute the linear model for the spectrum of a single pixel.
    Parameters:
    - basis: 2D array of basis functions, shape (n_freqs, n_basis)
    - coefficients: 1D array of fit coefficients, shape (n_basis,)
    Returns:
    - model: 1D array of model values, shape (n_freqs,)
    """
    model = basis @ coefficients
    return model

def tensor_prod_basis(moment_basis, BCSVD):
    """
    Compute the tensor product basis for the spectral moment basis.
    Parameters:
    - moment_basis: 2D array of spectral moment basis functions, shape (n_freqs, n_moments)
    - BCSVD: 2D array of beam chromaticity corrections, shape (n_freqs, n_beam)
    Returns:
    - tensor_basis: 2D array of tensor product basis functions, shape (n_freqs, n_moments*n_beam)
    """
    tensor_3d = moment_basis[:, :, np.newaxis] * BCSVD[:, np.newaxis, :]
    
    # Reshape to (n_freqs, n_moments * n_beam)
    n_freqs, n_moments = moment_basis.shape
    n_beam = BCSVD.shape[1]
    tensor_basis = tensor_3d.reshape((n_freqs, n_moments * n_beam))
    
    return tensor_basis


class fg_moment_basis:
    def __init__(self, freqs, nu_ref=None):  # Removed trailing comma
        """Initialize spectral moment basis generator.
        
        Args:
            freqs: Array of observation frequencies [MHz]
            nu_ref: Reference frequency [MHz]
        """

        if nu_ref is None:
            # let xs be the centred log freqs
            self.log_nu_ref = np.log(freqs).mean()
            self.log_xs = np.log(freqs) - self.log_nu_ref
        else:
            self.log_xs = np.log(freqs / nu_ref)  # Frequency ratios ν/ν_ref

    def basis(self, beta0, max_order, remove_1st_moment=False, BCSVD=None, BCF=None):
        """Generate orthogonalized spectral basis functions.
        
        Implements φ_k(ν) = (ν/ν_ref)^β₀ [ln(ν/ν_ref)]^k for k=0, ..., max_order
        
        Args:
            beta0: Reference spectral index (dimensionless)
            max_order: Maximum order of spectral moments to model
            remove_1st_moment: Whether to remove the first moment from the basis.
                If True, the first moment is removed from the basis.
                This is the case of adapted pivot.
            BCSVD: (nfreqs, nbeam) array of beam chromaticity corrections
            BCF: (nfreqs, ) array of beam chromaticity corrections

        Returns:
            basis_matrix: (n_freqs, n_moments) array of basis vectors
        """
        xs_beta0 = np.exp(self.log_xs*beta0)
        
        # Use Vandermonde matrix for stable polynomial basis
        vander = np.vander(self.log_xs, N=max_order+1, increasing=True)
        result = vander * xs_beta0[:, np.newaxis]

        if remove_1st_moment:
            # Remove the second column (i.e., the first moment)
            result = np.delete(result, 1, axis=1)

        # Apply Beam chromaticity corrections: first try BCSVD, if not provided, then BCF
        if BCSVD is not None:
            result = tensor_prod_basis(result, BCSVD)
        elif BCF is not None:
            result *= BCF[:, np.newaxis]

        # Normalize each column to have unit l2-norm
        norms = np.linalg.norm(result, axis=0)
        result /= norms[np.newaxis, :]  # Normalize each column by its l2-norm

        return result

    def fit_data_with_moments(self, 
                            data, 
                            beta_value, 
                            max_order, 
                            adaptive_pivot=True,
                            BCSVD=None, 
                            BCF=None,
                            return_loss=True):
        """
        Perform SED fit for a single pixel.
        """

        aux_basis = self.basis(beta_value,  max_order, remove_1st_moment=adaptive_pivot, BCSVD=BCSVD, BCF=BCF)

        if adaptive_pivot:
            # Fit the data with the adapted pivot
            
            def loss_func(params):
                beta = params[0]
                coefficients = params[1:]
                basis = self.basis(beta,  max_order, remove_1st_moment=adaptive_pivot, BCSVD=BCSVD, BCF=BCF)
                model_SED = linear_model(basis, coefficients)
                # define the loss function as the squares of the residuals
                residuals = (data - model_SED) / data
                loss = np.sqrt(np.mean(residuals**2))
                return loss

            # Minimize the loss function using scipy.optimize.minimize
            init_coeffs = linear_fit_1D(data, 
                                        aux_basis, 
                                        return_loss=False)
            initial_guess = [beta_value] + list(init_coeffs)
            result = minimize(loss_func, initial_guess, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            values = result.x
            if return_loss:
                loss = loss_func(values)
                return values, loss
            else:
                return values
    
        else:
            # Fit the data with the fixed pivot
            if return_loss:
                coefficients, loss = linear_fit_1D(data, aux_basis, return_loss=True)
                return coefficients, loss
            else:
                coefficients = linear_fit_1D(data, aux_basis, return_loss=False)
                return coefficients
        
    def null_space_basis(self, beta0, max_order, adaptive_pivot=False, BCF=None, BCSVD=None):

        matrix = self.basis(beta0, max_order, remove_1st_moment=adaptive_pivot, BCF=BCF, BCSVD=BCSVD)

        # Get orthonormal basis using modified Gram-Schmidt
        # ortho_basis = orth(matrix, rcond=1e-10)
        # proj = np.identity(ortho_basis.shape[0]) - ortho_basis @ ortho_basis.T
        ortho_basis = null_space(matrix.T, rcond=1e-10)
        return ortho_basis



        
      
class Global_21cm_extraction(fg_moment_basis):
    """
    Perform joint SED fit for foreground and cosmological 21cm signal.
    """

    def variance_in_fg_null_space(self, Cov_mat, beta0, max_order, adaptive_pivot=False, BCF=None, BCSVD=None):
        """
        Compute the covariance matrix projection in the null space of the spectral moment basis.
        """
        ortho_basis = self.null_space_basis(beta0, max_order, adaptive_pivot=adaptive_pivot, BCF=BCF, BCSVD=BCSVD)
        proj = ortho_basis @ ortho_basis.T 

        result = proj @ Cov_mat @ proj.T
        return np.sqrt(np.trace(result)/np.trace(Cov_mat))
    
    def joint_fg_21_fit(self, 
                        data, 
                        signal_basis,
                        beta_value, 
                        max_order, 
                        adaptive_pivot=True,
                        BCSVD=None, 
                        BCF=None,
                        return_loss=True):
        """
        Perform SED fit for a single pixel.
        Args:
            data: 1D array of data, shape (n_freqs,)
            signal_basis: 2D array of signal basis functions, shape (n_freqs, n_signal_modes)
            beta_value: Reference spectral index (dimensionless)
        """

        aux_basis = self.basis(beta_value,  max_order, remove_1st_moment=adaptive_pivot, BCSVD=BCSVD, BCF=BCF)
        aux_basis = np.hstack((aux_basis, signal_basis))
        n_signal_modes = signal_basis.shape[1]

        if adaptive_pivot:
            # Fit the data with the adapted pivot
            
            def loss_func(params):
                beta = params[0]
                coefficients = params[1:]
                fg_basis = self.basis(beta,  max_order, remove_1st_moment=adaptive_pivot, BCSVD=BCSVD, BCF=BCF)
                basis = np.hstack((fg_basis, signal_basis))
                model_SED = linear_model(basis, coefficients)
                # define the loss function as the squares of the residuals
                residuals = (data - model_SED) / data
                loss = np.sqrt(np.mean(residuals**2))
                return loss

            # Minimize the loss function using scipy.optimize.minimize
            init_coeffs = linear_fit_1D(data, 
                                        aux_basis, 
                                        return_loss=False)
            initial_guess = [beta_value] + list(init_coeffs)
            result = minimize(loss_func, initial_guess, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            values = result.x
            if return_loss:
                loss = loss_func(values)
                return values, loss
            else:
                return values
    
        else:
            # Fit the data with the fixed pivot
            if return_loss:
                coefficients, loss = linear_fit_1D(data, aux_basis, return_loss=True)
                return coefficients, loss
            else:
                coefficients = linear_fit_1D(data, aux_basis, return_loss=False)
                return coefficients
        






def fit_fg_cube(data_cube, spectral_index_map, freqs, nu_ref=None, max_order=5, adaptive_pivot=True, BCSVD=None, BCF=None):
    """
    Perform per-pixel spectral fitting across entire map.
    
    Parameters:
    - data_cube: (npix, nfreqs) array of spectral measurements
    - spectral_index_map: (npix,) array of spectral indices (beta0 values)
    - freqs: Array of observation frequencies [MHz]
    - nu_ref: Reference frequency [MHz]
    - n_moments: Number of spectral moments to fit
    
    Returns:
    - coeff_map: (npix, n_moments) array of coefficients
    - loss_map: (npix,) array of fractional MSE values
    """
    npix, nfreqs = data_cube.shape
    n_coeffs = max_order + 1
    if adaptive_pivot:
        n_coeffs -= 1
        pivot_map = np.zeros(npix)
    if BCSVD is not None:
        n_coeffs *= BCSVD.shape[1]
    
    coeff_map = np.zeros((npix, n_coeffs))
    loss_map = np.zeros(npix)
    
    # Precompute basis generator
    basis_gen = fg_moment_basis(freqs=freqs, nu_ref=nu_ref)

    if BCSVD is None and BCF is not None:
        results = Parallel(n_jobs=-1, verbose=0)(
                delayed(basis_gen.fit_data_with_moments)(
                    data_cube[i],
                    spectral_index_map[i],
                    max_order,
                    adaptive_pivot=adaptive_pivot,
                    BCF=BCF[i],
                    return_loss=True
                )
                for i in tqdm(range(npix), desc="Processing pixels", leave=False, dynamic_ncols=True)
            )
    else:
        results = Parallel(n_jobs=-1, verbose=0)(
                delayed(basis_gen.fit_data_with_moments)(
                    data_cube[i],
                    spectral_index_map[i],
                    max_order,
                    adaptive_pivot=adaptive_pivot,
                    BCSVD=BCSVD,
                    return_loss=True
                )
                for i in tqdm(range(npix), desc="Processing pixels", leave=False, dynamic_ncols=True)
            )
        
    # Unpack results
    for i in range(npix):
        if adaptive_pivot:
            aux, loss_map[i] = results[i]
            pivot_map[i] = aux[0]
            coeff_map[i] = aux[1:]
        else:
            coeff_map[i], loss_map[i] = results[i]
    if adaptive_pivot:
        return coeff_map, loss_map, pivot_map
    else:
        return coeff_map, loss_map



def fit_fg_plus_21_cube(data_cube, signal_basis, spectral_index_map, freqs, nu_ref=None, max_order=5, adaptive_pivot=True, BCSVD=None, BCF=None):
    """
    Perform per-pixel spectral fitting across entire map.
    
    Parameters:
    - data_cube: (npix, nfreqs) array of spectral measurements
    - spectral_index_map: (npix,) array of spectral indices (beta0 values)
    - freqs: Array of observation frequencies [MHz]
    - nu_ref: Reference frequency [MHz]
    - n_moments: Number of spectral moments to fit
    
    Returns:
    - coeff_map: (npix, n_moments) array of coefficients
    - loss_map: (npix,) array of fractional MSE values
    """
    npix, nfreqs = data_cube.shape
    n_coeffs = max_order + 1
    n_signal_modes = signal_basis.shape[1]
    if adaptive_pivot:
        n_coeffs -= 1
        pivot_map = np.zeros(npix)
        
    if BCSVD is not None:
        n_coeffs *= BCSVD.shape[1]
    coeff_map = np.zeros((npix, n_coeffs + n_signal_modes))
    loss_map = np.zeros(npix)
    
    # Precompute basis generator
    basis_gen = Global_21cm_extraction(freqs=freqs, nu_ref=nu_ref)

    if BCSVD is None and BCF is not None:
        results = Parallel(n_jobs=-1, verbose=0)(
                delayed(basis_gen.joint_fg_21_fit)(
                    data_cube[i],
                    signal_basis,
                    spectral_index_map[i],
                    max_order,
                    adaptive_pivot=adaptive_pivot,
                    BCF=BCF[i],
                    return_loss=True
                )
                for i in tqdm(range(npix), desc="Processing pixels", leave=False, dynamic_ncols=True)
            )
    else:
        results = Parallel(n_jobs=-1, verbose=0)(
                delayed(basis_gen.joint_fg_21_fit)(
                    data_cube[i],
                    signal_basis,
                    spectral_index_map[i],
                    max_order,
                    adaptive_pivot=adaptive_pivot,
                    BCSVD=BCSVD,
                    return_loss=True
                )
                for i in tqdm(range(npix), desc="Processing pixels", leave=False, dynamic_ncols=True)
            )
        
    # Unpack results
    for i in range(npix):
        if adaptive_pivot:
            aux, loss_map[i] = results[i]
            coeff_map[i] = aux[1:]
            pivot_map[i] = aux[0]
        else:
            coeff_map[i], loss_map[i] = results[i]
    
    if adaptive_pivot:
        return coeff_map, loss_map, pivot_map
    else:
        return coeff_map, loss_map


