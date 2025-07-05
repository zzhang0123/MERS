import glob
import numpy as np
import healpy as hp
from pyuvdata import UVData, UVCal, GaussianBeam, AiryBeam
import matplotlib.pyplot as plt



class RhinoBeam:

    def __init__(self, filepath='/your/file/path/'):
        # read and sort filenames in filepath
        self.filenames = sorted(glob.glob(filepath + '/*.fits'))
        self.nside = hp.get_nside(hp.read_map(self.filenames[0]))

    def get_beam_l(self, normalize=True):
        # Derive the beam transfer function bl as the sqrt of the power spectrum (Cl) of the beam map. 
        # This effectively take the azimuthal average of the beam map
        bl_list = []
        for filename in self.filenames:
            beam = hp.read_map(filename)
            # Compute the power spectrum (Cl) of the beam map
            cl_beam = hp.anafast(beam, lmax=3*self.nside)
            # The beam transfer function bl is the square root of Cl, normalized to bl[0]=1
            bl = np.sqrt(cl_beam)
            if normalize:
                bl /= bl[0]
            bl_list.append(bl)
        self.bl_list = bl_list 
        pass

    def generate_BCF(self, ref_map, ref_beam_window):
        # Convolve the ref_map with the reference beam transfer function
        ref_map_convolved = hp.smoothing(ref_map, beam_window=ref_beam_window)
        # Convolve the ref_map with the beam transfer functions
        BCF_cube = [hp.smoothing(ref_map, beam_window=beam_window) / ref_map_convolved
                            for beam_window in self.bl_list]
        return np.array(BCF_cube).T   # Shape: (npix, nfreq)

    def generate_SVD_beams(self):
        '''
        Perform SVD on the beam array.

        Returns:
        U: Unitary matrix having left singular vectors as columns.
            $U[:, i] = s_i(\nu)$
        S: The singular values, sorted in non-increasing order.
        Vt: Unitary matrix having right singular vectors as rows.
            $Vt[i, :] = B_i(p)$
        '''
        beam_list=[]
        for filename in self.filenames:
            beam = hp.read_map(filename)
            beam_list.append(beam)
        beam_list = np.array(beam_list) # Shape: (nfreq, npix)
        # Perform SVD on the beam_list
        U, S, Vt = np.linalg.svd(beam_list, full_matrices=False)
        return U, S, Vt


class Achromatic_Gaussian_Beam(RhinoBeam):
    def __init__(self, fwhm_deg, n_freqs, nside=512):
        self.nside = nside
        self.fwhm_deg = fwhm_deg
        self.n_freqs = n_freqs

        pyuv_obj = GaussianBeam(sigma = np.deg2rad(self.fwhm_deg/2.355),
                                sigma_type='power')
        gaussian_beam = pyuv_obj.to_uvbeam(freq_array=np.array([70e6]),
                                           beam_type='power', 
                                           pixel_coordinate_system='healpix', 
                                           nside=self.nside)
        gaussian_beam = np.abs(gaussian_beam.data_array[0,0,0])
        
        self.beams = [gaussian_beam for i in range(n_freqs)]

        pass

    def get_beam_l(self, normalize=True):
        bl_list = []
        for beam in self.beams:
            cl_beam = hp.anafast(beam, lmax=3*self.nside)
            bl = np.sqrt(cl_beam)
            if normalize:
                bl /= bl[0]
            bl_list.append(bl)
        self.bl_list = bl_list
        pass

    def generate_SVD_beams(self):
        beam_list = np.array(self.beams)
        # Perform SVD on the beam_list
        U, S, Vt = np.linalg.svd(beam_list, full_matrices=False)
        return U, S, Vt
    

    
class Circular_Aperture_Gaussian_Envolope(Achromatic_Gaussian_Beam):
    def __init__(self, first_null_200_mhz_deg, freq_array_mhz=None, nside=512, gaussian_envelope_sigma=None):
        aperture_diameter = (1.22 * 299792458) / (np.sin(np.deg2rad(first_null_200_mhz_deg)) * 200e6)
        self.nside = nside

        if gaussian_envelope_sigma is None:
            gaussian_envelope_sigma = 2*first_null_200_mhz_deg
        
        pyuv_obj = GaussianBeam(sigma = np.deg2rad(gaussian_envelope_sigma),
                                sigma_type='power')
        gaussian_beam = pyuv_obj.to_uvbeam(freq_array=np.array([70e6]),
                                           beam_type='power', 
                                           pixel_coordinate_system='healpix', 
                                           nside=self.nside)
        gaussian_beam = np.abs(gaussian_beam.data_array[0,0,0])

        analytic_obj = AiryBeam(diameter = aperture_diameter)
        
        if freq_array_mhz is None:
            freq_array_mhz = np.linspace(50, 200, 10)
        
        uvbeam_object = analytic_obj.to_uvbeam(freq_array = freq_array_mhz*10**6,
                                               beam_type='power',
                                               pixel_coordinate_system='healpix',
                                               nside=self.nside)
        
        data_array = np.abs(uvbeam_object.data_array)
        del uvbeam_object
        data_array = data_array[0,0]

        self.beams = [b*gaussian_beam for b in data_array]



from matplotlib.colors import SymLogNorm

 
def plot_svd_beam_modes(Vt, nest=False, figsize=(12, 8)):
    """
    Plots the first four SVD beam modes as HEALPix maps in a 2x2 grid with:
    - North Pole rotated to center
    - Common color bar for all subplots
    
    Parameters:
    Vt (ndarray): V^T matrix from SVD (shape: [k, npix])
    nside (int): HEALPix resolution parameter
    nest (bool): If True, assumes NESTED ordering (default: RING)
    figsize (tuple): Figure size (width, height) in inches
    """
    if Vt.shape[0] < 4:
        raise ValueError("Vt must have at least 4 rows (modes)")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Find global min/max for common color scaling
    vmax = max(np.max(np.abs(mode)) for mode in Vt[:4])
    vmin = -vmax

    # Calculate scaling parameters
    linthresh = 0.1*vmax  # Threshold for linear scale near zero
    linscale = 0.5        # Transition smoothness

    # Replace norm in previous code with:
    norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)

    # # Calculate percentiles (adjust 5th/95th percentiles as needed)
    # p_low = np.percentile(np.concatenate(Vt[:4]), 5)
    # p_high = np.percentile(np.concatenate(Vt[:4]), 95)
    # vmax = max(abs(p_low), abs(p_high))
    # vmin = -vmax
    # # Use linear scaling between percentiles
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each mode in its own subplot
    images = []
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, projection='mollweide')
        # Plot HEALPix map with rotation to center North Pole
        if Vt[i-1, 0] < 0:
            beam_mode = -Vt[i-1]
        else:
            beam_mode = Vt[i-1]
        img = hp.mollview(
            beam_mode,
            nest=nest,
            rot=(0, 90),  # Center North Pole (0° lon, 90° lat)
            title=f'Mode {i}',
            cmap='RdBu_r',
            notext=True,   # No coordinate text
            min=vmin,
            max=vmax,
            #norm=norm,
            return_projected_map=True,
            hold=True,
            fig=fig,
            sub=(2, 2, i)
        )
        images.append(img)
    
    # # Add common color bar at bottom
    # cax = fig.add_axes([0.2, 0.05, 0.65, 0.02])  # [left, bottom, width, height]
    # cbar = ColorbarBase(cax, cmap=plt.cm.RdBu_r, norm=norm, 
    #                             orientation='horizontal')
    # cbar.set_label('Amplitude')
    
    # Adjust layout and show
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.15)  # Make space for color bar
    plt.show()

# -------

def gaussian(x, amp, sigma, mu=0):
    return amp * np.exp(- ((x-mu)**2) / (2 * sigma**2))

def sinc_squared(x, fwhm):
    a = 5.57 / fwhm
    return (np.sin(a*x/2) / (a* x / 2))**2

class Achromatic_Gaussian(RhinoBeam):
    def __init__(self, nfreqs, fwhm_deg=10, nside=512):
        npix = 12 * (nside**2)
        self.nside=nside
        theta, _ = hp.pix2ang(nside, np.arange(npix)) # set up thetas for gaussian function
        sigma_rad = np.deg2rad(fwhm_deg) / (2*np.sqrt(2*np.log(2))) 
        beam = gaussian(theta, amp=1, sigma=sigma_rad, mu=0)
        beam  = beam / np.sum(beam) # normalise beam
        self.beams = [beam for i in range(nfreqs)]
        pass
    
    def get_beam_l(self, normalize=True):
        bl_list = []
        for beam in self.beams:
            cl_beam = hp.anafast(beam, lmax=3*self.nside)
            bl = np.sqrt(cl_beam)
            if normalize:
                bl /= bl[0]
            bl_list.append(bl)
        self.bl_list = bl_list
        pass

    def generate_SVD_beams(self):
        beam_list = np.array(self.beams)
        # Perform SVD on the beam_list
        U, S, Vt = np.linalg.svd(beam_list, full_matrices=False)
        return U, S, Vt

class Achromatic_Sinc_Squared(Achromatic_Gaussian):
    def __init__(self, nfreqs, fwhm_deg=10, nside=512):
        npix = 12 * (nside**2)
        self.nside = nside
        theta, _ = hp.pix2ang(nside, np.arange(npix)) # set up thetas for gaussian function

        beam = sinc_squared(theta, fwhm=np.deg2rad(fwhm_deg))
        beam  = beam / np.sum(beam) # normalise beam
        self.beams = [beam for i in range(nfreqs)]
        pass

class Chromatic_Gaussian(Achromatic_Gaussian):
    def __init__(self, freq_list_mhz, fwhm_function, fwhm_func_params, nside=512):
        fwhms = [fwhm_function(nu, fwhm_func_params) for nu in freq_list_mhz] # get fwhm from input function
        npix = 12 * (nside**2)
        self.nside = nside
        theta, _ = hp.pix2ang(nside=nside, ipix=np.arange(npix))

        beam_list = []
        for f in fwhms:
            sigma_rad = np.deg2rad(f) / (2*np.sqrt(2*np.log(2))) 
            beam = gaussian(theta, amp=1, sigma=sigma_rad)
            beam = beam / np.sum(beam)
            beam_list.append(beam)
        self.beams = beam_list
        pass

class Chromatic_Sinc_Squared(Achromatic_Gaussian):
    def __init__(self, freq_list_mhz, fwhm_function, fwhm_func_params, nside=512):
        fwhms = [fwhm_function(nu, fwhm_func_params) for nu in freq_list_mhz] # get fwhm from input function
        npix = 12 * (nside**2)
        self.nside = nside
        theta, _ = hp.pix2ang(nside=nside, ipix=np.arange(npix))

        beam_list = []
        for f in fwhms:
            beam = sinc_squared(theta, fwhm=np.deg2rad(f))
            beam = beam / np.sum(beam)
            beam_list.append(beam)
        self.beams = beam_list
        pass
