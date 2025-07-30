import numpy as np
import healpy as hp
from scipy.special import sph_harm_y

from sympy import lambdify, symbols
from sympy.physics.quantum.spin import Rotation
from scipy.linalg import cho_factor, cho_solve


def real_sph_harm(l, m, theta, phi):
    """
    Evaluate real spherical harmonic Y^{real}_{l m}(theta, phi) at scalar or array input.

    Parameters:
        l (int): degree
        m (int): order
        theta (array): colatitude angle(s), [0, pi]
        phi (array): azimuthal angle(s), [0, 2pi]

    Returns:
        Real spherical harmonic evaluated at (theta, phi), shape compatible with input.
    """
    if m < 0:
        return np.sqrt(2) * sph_harm_y(l, -m, theta, phi).imag
    elif m == 0:
        return sph_harm_y(l, 0, theta, phi).real
    else:
        return np.sqrt(2) * sph_harm_y(l, m, theta, phi).real

def build_design_matrix(theta, phi, B_l0, lmax):
    """
    Build design matrix A for system:
        d(n, nu) = sum_{lm} sqrt(4pi / (2l+1)) * Y_lm(n) * B_l0 * T_lm(nu)

    Inputs:
        theta, phi : arrays of shape (Nsky,)
        B_l0       : array of shape (lmax+1,) – beam profile for each l
        lmax       : maximum multipole

    Returns:
        A         : ndarray (Nsky, N_modes) – design matrix
        lm_list   : list of (l, m) tuples corresponding to each column
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    A_cols = []
    lm_list = []

    for l in range(lmax + 1):
        norm_factor = np.sqrt(4 * np.pi / (2 * l + 1))
        for m in range(-l, l + 1):
            Y_lm_real = real_sph_harm(l, m, theta, phi)  # (N,)
            column = norm_factor * B_l0[l] * Y_lm_real
            A_cols.append(column)
            lm_list.append((l, m))

    A = np.vstack(A_cols).T  # Shape: (Nsky, N_modes)
    return A, lm_list



def maximum_likelihood_filter(A, d, Ninv):
    """
    Solve (A^T N^{-1} A) x = A^T N^{-1} d
    """
    At_Ninv = A.T @ Ninv
    lhs = At_Ninv @ A
    rhs = At_Ninv @ d

    # Use Cholesky for numerical stability
    c, low = cho_factor(lhs)
    x_ml = cho_solve((c, low), rhs)
    return x_ml

def solve_modes(A, d, Ninv=None):
    """
    Solve for spherical harmonic coefficients from linear system A @ T = d.
    
    Parameters:
        A : (Ndata, N_modes) design matrix
        d : (Ndata,) data vector
        Ninv : (Ndata, Ndata) optional inverse noise covariance matrix

    Returns:
        T_hat : (N_modes,) estimated SH coefficients
    """
    if Ninv is None:
        T_hat, residuals, rank, s = np.linalg.lstsq(A, d, rcond=None)
        return T_hat
    else:
        # If Ninv is provided, use maximum likelihood estimation
        return maximum_likelihood_filter(A, d, Ninv)

def complex_to_real_alm(a_complex, lmax):
    """Convert complex alm to real spherical harmonic coefficients (real-valued basis).
    
    Parameters:
        a_complex : dict mapping (l, m) → complex coefficients (0 <= m <= l)
        lmax      : maximum l
        
    Return:
        a_real : dict mapping (l, m) → real coefficients (-l <= m <= l)
    """
    a_real = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if m < 0:
                a_real[(l, m)] = - np.sqrt(2) * a_complex[(l, -m)].imag
            elif m == 0:
                a_real[(l, m)] = a_complex[(l, 0)].real
            else:
                a_real[(l, m)] = np.sqrt(2) * a_complex[(l, m)].real
    return a_real

def real_to_complex_alm(a_real, lmax):
    """Inverse of the custom complex_to_real_alm: convert real to complex spherical harmonic coefficients.

    Parameters:
        a_real : dict mapping (l, m) → real coefficients (-l <= m <= l)
        lmax   : maximum l

    Returns:
        a_complex : dict mapping (l, m) → complex coefficients (0 <= m <= l)
    """
    a_complex = {}
    for l in range(lmax + 1):
        # m = 0
        a_complex[(l, 0)] = a_real[(l, 0)] + 0j
        for m in range(1, l + 1):
            R = a_real[(l, m)] / np.sqrt(2) 
            I = - a_real[(l, -m)] / np.sqrt(2)   # note: (-1)^{-m} = (-1)^m
            a_complex[(l, m)] = R + 1j * I
            a_complex[(l, -m)] = (R - 1j * I) * (-1)**m
    return a_complex

def real_array2dict(a_real_arr, lmax, lm_list=None):
    """
    Convert a real-valued array of spherical harmonic coefficients to a dictionary format.
    """
    if lm_list is None:
        lm_list = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    a_real_dict = {lm: coeff for lm, coeff in zip(lm_list, a_real_arr)}
    return a_real_dict

def complex_array2dict(a_complex_arr):
    """
    Convert a complex-valued array of spherical harmonic coefficients to a dictionary format.
    """
    num = len(a_complex_arr)
    lmax = hp.Alm.getlmax(num)
    a_complex_dict = { hp.Alm.getlm(lmax, i): a_complex_arr[i] for i in np.arange(num) }
    return a_complex_dict
    

def convert_real_to_complex_alm(a_real_arr, lmax):
    """
    Convert real-valued spherical harmonic coefficients array to full healpy complex alm array.
    
    Parameters:
        a_real_arr : array mapping (l, m) → real coefficients (-l <= m <= l)
        lmax        : maximum l

    Returns:
        alm_array : complex alm array of size hp.Alm.getsize(lmax)
    """
    a_real_dict = real_array2dict(a_real_arr, lmax)
    a_complex_dict = real_to_complex_alm(a_real_dict, lmax)

    # Build alm array in healpy format
    alm_arr = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    for (l, m), val in a_complex_dict.items():
        idx = hp.Alm.getidx(lmax, l, abs(m))
        if m >= 0:
            alm_arr[idx] = val
    return alm_arr

def test_conversion():
    lmax = 5
    np.random.seed(42)

    a_complex_in = {}
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            val = np.random.randn() + 1j * np.random.randn()
            a_complex_in[(l, m)] = val
            if m != 0:
                a_complex_in[(l, -m)] = (-1)**m * np.conj(val)
            else:
                a_complex_in[(l, 0)] = val.real

    # complex to real conversion
    a_real = complex_to_real_alm(a_complex_in, lmax)


    a_complex_out = real_to_complex_alm(a_real, lmax)

    # real to complex conversion
    errors = []
    for key in a_complex_in:
        if not np.allclose(a_complex_in[key], a_complex_out[key], atol=1e-12):
            errors.append((key, a_complex_in[key], a_complex_out[key]))

    if not errors:
        print("✅ Passed: Conversion from complex → real → complex is consistent.")
    else:
        print("❌ Failed:")
        for key, a_in, a_out in errors:
            print(f"  {key}: in={a_in}, out={a_out}, diff={a_in - a_out}")
    
    pass



def evaluate_sky(theta, phi, T_lm, lmax, B_l0=None, complex_basis=False):
    """
    Evaluate sky temperature at given positions using real SH basis.

    Parameters:
        theta, phi : directions (arrays of shape (N,))
        T_lm       : array of SH coefficients (real)
        lm_list    : list of (l, m) tuples
        B_l0       : optional array (lmax+1,) for convolution

    Returns:
        T(theta, phi) : array (N,) sky signal at those points
    """
    
    T = np.zeros_like(theta)

    if complex_basis:
        T_lm = complex_array2dict(T_lm) 
        T_lm = complex_to_real_alm(T_lm, lmax)
    else:
        assert len(T_lm) == (lmax+1)**2
        T_lm = real_array2dict(T_lm, lmax)
    
    for l in range(lmax + 1):
        norm = np.sqrt(4 * np.pi / (2 * l + 1))
        beam = B_l0[l] if B_l0 is not None else 1.0
        bl = norm * beam
        for m in range(-l, l + 1):
            coeff = T_lm[(l, m)]
            Ylm_real = real_sph_harm(l, m, theta, phi)
            T += coeff * bl * Ylm_real
    return T



#########################################################################
# Below are functions for build design matrix using complex spherical harmonics
# (Not used in the paper. But useful for scenarios with complex SH)
#########################################################################

def build_design_matrix_complex_sph_harm(thetas, phis, B_l0, lmax):
    """
    Returns real design matrix A and list of (l, m, kind)
    where kind ∈ {"m0", "re", "im"}
    """

    phi_sym, theta_sym = symbols('ϕ θ', real=True)

    A_cols = []
    lm_labels = []

    for l in range(lmax + 1):
        for m in range(l + 1):  # only m ≥ 0
            # Create D_{m0}^l(ϕ, θ, 0)^*
            norm = np.sqrt((4 * np.pi) / (2 * l + 1))
            D_expr = norm * Rotation.D(l, m, 0, phi_sym, theta_sym, 0).doit().conjugate()
            D_func = lambdify((phi_sym, theta_sym), D_expr, modules='numpy')

            D_vals = B_l0[l] * D_func(phis, thetas)  # shape = (N_samples,)
            if D_vals.ndim == 0: # this is the case of l=0 and m=0.
                D_vals *= np.ones(len(thetas))  # ensure shape is (N_samples,)

            if m == 0:
                A_cols.append(D_vals.real)
                lm_labels.append((l, 0, 'm0'))
            else:
                A_cols.append(D_vals.real)
                A_cols.append(D_vals.imag)
                lm_labels.append((l, m, 're'))
                lm_labels.append((l, m, 'im'))

    A = np.vstack(A_cols).T
    return A, lm_labels




def real_to_alm_comp(x, lm_labels, lmax):
    """
    Reconstruct T_{lm} complex coefficients from realified x vector.
    Returns: complex alm array usable by healpy 
    """
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)

    T_lm = {}  # temporary dictionary
    for coeff, (l, m, part) in zip(x, lm_labels):
        if part == "m0":
            T_lm[(l, 0)] = coeff
        elif part == "re":
            T_lm[(l, m)] = coeff
        elif part == "im":
            T_lm[(l, m)] = T_lm.get((l, m), 0) + 1j * coeff

    # Use symmetry T_{l,-m} = (-1)^m T_{lm}^*
    for l in range(lmax + 1):
        for m in range(l + 1):
            if m == 0:
                alm[hp.Alm.getidx(lmax, l, 0)] = T_lm[(l, 0)].real
            else:
                alm[hp.Alm.getidx(lmax, l, m)] = T_lm.get((l, m), 0)

    return alm

def realify_alm_comp(alm, lmax):
    """
    Convert complex alm to realified vector x.
    Returns: x vector and lm_labels
    """
    lm_labels = []
    x = []

    for l in range(lmax + 1):
        for m in range(l + 1):  # only m ≥ 0
            if m == 0:
                x.append(alm[hp.Alm.getidx(lmax, l, 0)].real)
                lm_labels.append((l, 0, 'm0'))
            else:
                x.append(alm[hp.Alm.getidx(lmax, l, m)].real)
                x.append(alm[hp.Alm.getidx(lmax, l, m)].imag)
                lm_labels.append((l, m, 're'))
                lm_labels.append((l, m, 'im'))

    x = np.array(x)

    return x, lm_labels

