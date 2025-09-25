import healpy as hp
import numpy as np
from pygdsm import GlobalSkyModel
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os

filepath = os.path.dirname(__file__)
haslam_path = os.path.join(filepath, 'data/haslam408_dsds_Remazeilles2014.fits')
cnnpl_path = os.path.join(filepath, 'data/cnn56arcmin_beta.npy')
EM_path = os.path.join(filepath, 'data/EM_mean_std.fits')
COM_path = os.path.join(filepath, 'data/COM_CompMap_freefree-commander_0256_R2.00.fits')
GLEAM_path_408 = os.path.join(filepath, "data/gleam_nside512_K_allsky_408MHz.npy")
# GLEAM_path_16 = os.path.join(filepath, "data/gleam_nside512_K_allsky_50MHz_16freqs.npz")


def smoothed_maps(maps, beam_transfer):
    """
    Smooth the maps with the given beam transfer function
    - maps: array with shape (npix, nfreq)
    - beam_transfer: array with shape (nfreq, lmax)
    Returns: 
    - smoothed_map: array with shape (npix, nfreq)
    """
    smoothed_map = np.zeros_like(maps)
    for i, bl in enumerate(beam_transfer):
        smoothed_map[:, i] = hp.smoothing(maps[:, i], beam_window=bl)
    return smoothed_map

def change_nside(map, nside_out):
    """
    Change the nside of a map
    - map: input map
    - nside_out: desired nside
    Returns: map with the new nside
    """
    nside_old = hp.get_nside(map)
    if nside_old != nside_out:
        return hp.ud_grade(map, nside_out=nside_out)
    return map

class SynchrotronExtrapolator:
    def __init__(self, reference_map=None, spectral_index_map=None, reference_freq=408, nside=128):
        """
        Initialize with:
        - reference_map: Haslam 408 MHz map
        - spectral_index_map: the spectral index map
        - reference_freq: reference frequency in MHz (default: 408)
        """
        self.nside = nside
        if reference_map is None:
            self.reference_map = hp.read_map(haslam_path)
        else:
            self.reference_map = reference_map
        self.reference_map = change_nside(self.reference_map, nside)
        if spectral_index_map is None:
            self.specidx_map = np.load(cnnpl_path)
        else:
            self.specidx_map = spectral_index_map
        self.specidx_map = change_nside(self.specidx_map, nside)
        self.ref_freq = reference_freq

    def index_curvature(self, target_freqs):
        """
        Take the spectral index map between 0.408 and 23 GHz and move in to any frequency
        - target_freqs: scalar or 1d array/list of frequencies in MHz
        """

        # Ensure target_freqs is an array-like
        target_freqs = np.atleast_1d(target_freqs)
        ref_mhz = ((23000 - 408) / 2.0) + 408
        # c vaue taken from https://academic.oup.com/mnras/article/509/4/4923/6433658
        c = -0.10
        
        relevant_inds = np.array([self.specidx_map + c * np.log(target_freqs[ff]/ref_mhz) for ff in range(len(target_freqs))])

        return relevant_inds

    def beta_distribution(self, freq, beam_map, n_bins=100, rotation=None, show_hist=False):
        """
        Calculate the beta distribution of the spectral index map at a given frequency
        - n_bins: number of bins
        - beam_map: the beam transfer function
        - rotation: optional, if provided, should be a tuple of (lon, lat, phi) in degrees; 
                    For RHINO beam, should be [target_lon, 90 + target_lat, 0], as the healpix beam is pointing towards the south pole.
        Returns:
        - hist: the histogram of the spectral index map
        - bin_edges: the bin edges of the histogram
        """
        # Get the max and min beta values
        specind_map = self.index_curvature(np.array([freq]))[0,:]
        min_beta = np.min(specind_map)
        max_beta = np.max(specind_map)

        # Create bins
        bins = np.linspace(min_beta, max_beta, n_bins+1)

        if rotation is not None:
            rotator = hp.Rotator(rot=rotation, deg=True)
            # Rotate the 
            weight_map = rotator.rotate_map_pixel(beam_map) * self.reference_map
        else:
            weight_map = beam_map * self.reference_map

        # Calculate the histogram
        hist, bin_edges = np.histogram(specind_map, bins=bins, weights=weight_map)

        if show_hist:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], align='center')
            plt.xlabel('Spectral Index β')
            plt.ylabel('Weighted Count')
            plt.title('Beam-weighted Spectral Index Distribution')
            plt.show()
            return

        return hist, bin_edges 
        
    def map(self, target_freqs, beam_transfer=None):
        """
        Extrapolate to target frequency/frequencies and save map(s)
        - target_freqs: scalar or 1d array/list of frequencies in MHz
        - beam_transfer: optional, if provided, should be an array with shape (nfreq, lmax)
        Returns: array with shape (npix,) for scalar input or (npix, nfreq) for array input
        """
        if self.reference_map is None or self.specidx_map is None:
            raise ValueError("Maps must be loaded before extrapolation")

        # Ensure target_freqs is an array-like
        target_freqs = np.atleast_1d(target_freqs)

        # Vectorized calculation with proper broadcasting
        specind_map = self.index_curvature(target_freqs)
        pure_diffuse = self.reference_map - 8.9 # monopole (CMB+radio background) from https://arxiv.org/abs/1411.7616
        extrap_maps = np.array([(pure_diffuse * (target_freqs[ff]/self.ref_freq) ** specind_map[ff, :]) for ff in range(len(target_freqs))]).T # Shape: (npix, nfreq)

        # Apply smoothing if requested
        if beam_transfer is not None:
            return smoothed_maps(extrap_maps, beam_transfer)

        return extrap_maps if len(target_freqs) > 1 else extrap_maps.squeeze()
    
class FreeFreeExtrapolator:
    def __init__(self, em_map=None, etemp_map=None, nside=128):
        """
        Initialize with:
        - em_map: Hutchenrouter emission measure map
        - etemp_map: Planck electron temp map
        Both these maps upgraded from Nside 256 to 512
        """
        self.nside = nside
        if em_map is None:
            self.em_map = hp.ud_grade(hp.read_map(EM_path), nside_out=nside)
        else:
            self.em_map = hp.ud_grade(em_map, nside_out=nside)
        if etemp_map is None:
            self.etemp_map = hp.ud_grade(hp.read_map(COM_path, field=4), nside_out=nside)
        else:
            self.etemp_map = hp.ud_grade(etemp_map, nside_out=nside)
        
    def map(self, target_freqs, beam_transfer=None):
        """
        Extrapolate to target frequency/frequencies and save map(s)
        - target_freqs: scalar or 1d array/list of frequencies in MHz
        - beam_transfer: optional, if provided, should be an array with shape (nfreq, lmax)
        Returns: array with shape (npix,) for scalar input or (npix, nfreq) for array input
        """
        if self.etemp_map is None or self.em_map is None:
            raise ValueError("Maps must be loaded before extrapolation")

        # Ensure target_freqs is an array-like
        target_freqs = np.atleast_1d(target_freqs) / 1000. #freqs in GHz
        npix = hp.nside2npix(self.nside)    
        nfreqs = len(target_freqs)
        extrap_maps = np.zeros((npix, nfreqs))
        
        #free-free calculation from Table 4 of https://www.aanda.org/articles/aa/full_html/2016/10/aa25967-15/aa25967-15.html#T4
        for ff in range(nfreqs):
            fact = np.log(target_freqs[ff] * (self.etemp_map/1.e4)**(-3/2.))
            expfact = np.exp(5.960 - (np.sqrt(3) / np.pi) * fact)
            gaunt = np.log(expfact + np.exp(1))
            tau = 0.05468 * self.etemp_map**(-3/2.) * target_freqs[ff]**(-2.0) * self.em_map * gaunt
            extrap_maps[:, ff] = self.etemp_map * (1 - np.exp(-1. * tau))

        # Apply smoothing if requested
        if beam_transfer is not None:
            return smoothed_maps(extrap_maps, beam_transfer)

        return extrap_maps if len(target_freqs) > 1 else extrap_maps.squeeze()

def CNN_PL_sky(freq_list, beam_transfer=None, nside=128, return_spec_index=False):
    Mel_sync_model = SynchrotronExtrapolator(nside=nside)
    mel_sync = Mel_sync_model.map(freq_list, beam_transfer=beam_transfer)

    Mel_ff_model = FreeFreeExtrapolator(nside=nside)
    mel_ff = Mel_ff_model.map(freq_list, beam_transfer=beam_transfer)

    mel_diffuse = mel_sync + mel_ff
    if return_spec_index:
        freq_mid_ind = len(freq_list) // 2
        spec_index = Mel_sync_model.index_curvature(freq_list[freq_mid_ind])[0]
        return mel_diffuse, spec_index
    return mel_diffuse # shape: (npix, nfreq)

def GSM_maps(freqs, nside=128, beam_transfer=None):
    """Generate GSM maps for multiple frequencies"""
    # Initialize GSM with parameters
    gsm = GlobalSkyModel(freq_unit='MHz')

    # Generate maps for all frequencies
    maps = []
    for freq in np.atleast_1d(freqs):
        maps.append(gsm.generate(freq))
    
    # if len(maps) == 1:
    #     return maps[0]  # Return single map if only one frequency was provided
    # Stack maps along frequency axis
    maps = np.stack(maps, axis=1)  # Shape: (npix, nfreq)

    # Apply smoothing if requested
    if beam_transfer is not None:
        maps = smoothed_maps(maps, beam_transfer)

    # Initialize new map cube
    npix = hp.nside2npix(nside)
    map_cube = np.zeros((npix, maps.shape[1]))

    # Loop over each frequency/time slice
    for i in range(maps.shape[1]):
        map_cube[:, i] = hp.ud_grade(maps[:, i],
                                    nside_out=nside,
                                    power=2)  # power=2 for intensity maps

    if map_cube.shape[1] == 1:
        return map_cube.squeeze()  # Return single map if only one frequency was provided
    return map_cube  


# point source model with varying beta

# class ptsrc_interp():
#     def __init__(self, freq_list, filepath=GLEAM_path_16, nside=128, beam_transfer=None):
#         data = np.load(filepath)
#         self.freq_list = freq_list
#         self.psfg = np.array([
#                         hp.ud_grade(m, nside_out=nside)
#                         for m in data["psrc_sky"]
#                         ])
#         interp = RegularGridInterpolator((data["freqs"], np.arange(np.shape(self.psfg)[-1])), self.psfg)
#         X, Y = np.meshgrid(freq_list, np.arange(np.shape(self.psfg)[-1]))
#         maps = interp((X, Y)) # shape: (npix, nfreq)

#         if beam_transfer is not None:
#             self.maps = smoothed_maps(maps, beam_transfer)
#         else:
#             self.maps = maps


class ptsrc_powerlaw():
    def __init__(self, filepath=GLEAM_path_408, beta_psfg = -2.3, nside=128):
        self.beta_psfg = beta_psfg
        data = np.load(filepath)
        self.psfg = hp.ud_grade(data, nside_out=nside)

    def __call__(self, freq_list, beam_transfer=None):
        maps = np.outer(self.psfg, (freq_list/408)**self.beta_psfg)
        if beam_transfer is not None:
            return smoothed_maps(maps, beam_transfer)
        else:
            return maps
