import healpy as hp
import numpy as np
from pygdsm import GlobalSkyModel
import matplotlib.pyplot as plt
import glob
import os

filepath = os.path.dirname(__file__)
haslam_path = os.path.join(filepath, 'data/haslam408_dsds_Remazeilles2014.fits')
cnnpl_path = os.path.join(filepath, 'data/cnn56arcmin_beta.npy')
EM_path = os.path.join(filepath, 'data/EM_mean_std.fits')
COM_path = os.path.join(filepath, 'data/COM_CompMap_freefree-commander_0256_R2.00.fits')

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

class SynchrotronExtrapolator:
    def __init__(self, reference_map=None, spectral_index_map=None, reference_freq=408):
        """
        Initialize with:
        - reference_map: Haslam 408 MHz map
        - spectral_index_map: the spectral index map
        - reference_freq: reference frequency in MHz (default: 408)
        """
        if reference_map is None:
            self.reference_map = hp.read_map(haslam_path)
        else:
            self.reference_map = reference_map
        if spectral_index_map is None:
            self.specidx_map = np.load(cnnpl_path)
        else:
            self.specidx_map = spectral_index_map
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
    def __init__(self, em_map=None, etemp_map=None):
        """
        Initialize with:
        - em_map: Hutchenrouter emission measure map
        - etemp_map: Planck electron temp map
        Both these maps upgraded from Nside 256 to 512
        """
        if em_map is None:
            self.em_map = hp.ud_grade(hp.read_map(EM_path), 512)
        else:
            self.em_map = em_map
        if etemp_map is None:
            self.etemp_map = hp.ud_grade(hp.read_map(COM_path, field=4), 512)
        else:
            self.etemp_map = etemp_map
        
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
        npix = 12 * 512 * 512
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


def GSM_maps(freqs, nside=512, beam_transfer=None):
    """Generate GSM maps for multiple frequencies"""
    # Initialize GSM with parameters
    gsm = GlobalSkyModel(freq_unit='MHz')
    gsm.nside=nside
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

    if maps.shape[1] == 1:
        return maps.squeeze()  # Return single map if only one frequency was provided
    return maps   

