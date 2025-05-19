import healpy as hp
import numpy as np
from pygdsm import GlobalSkyModel
import matplotlib.pyplot as plt


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

class SynchrotronExtrapolator:
    def __init__(self, reference_map=None, spectral_index_map=None, reference_freq=408):
        """
        Initialize with:
        - reference_map: Haslam 408 MHz map
        - spectral_index_map: the spectral index map
        - reference_freq: reference frequency in MHz (default: 408)
        """
        if reference_map is None:
            self.reference_map = hp.read_map('haslam408_dsds_Remazeilles2014.fits')
        else:
            self.reference_map = reference_map
        if spectral_index_map is None:
            self.specidx_map = np.load('cnn56arcmin_beta.npy')
        else:
            self.specidx_map = spectral_index_map
        self.ref_freq = reference_freq

    def beta_distribution(self, beam_map, n_bins=100, rotation=None, show_hist=False):
        """
        Calculate the beta distribution of the spectral index map
        - n_bins: number of bins
        - beam_map: the beam transfer function
        - rotation: optional, if provided, should be a tuple of (lon, lat, phi) in degrees; 
                    For RHINO beam, should be [target_lon, 90 + target_lat, 0], as the healpix beam is pointing towards the south pole.
        Returns:
        - hist: the histogram of the spectral index map
        - bin_edges: the bin edges of the histogram
        """
        # Get the max and min beta values
        min_beta = np.min(self.specidx_map)
        max_beta = np.max(self.specidx_map)

        # Create bins
        bins = np.linspace(min_beta, max_beta, n_bins+1)

        if rotation is not None:
            rotator = hp.Rotator(rot=rotation, deg=True)
            # Rotate the 
            weight_map = rotator.rotate_map_pixel(beam_map) * self.reference_map
        else:
            weight_map = beam_map * self.reference_map

        # Calculate the histogram
        hist, bin_edges = np.histogram(self.specidx_map, bins=bins, weights=weight_map)

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
        scaling = (target_freqs[:, np.newaxis]/self.ref_freq) ** self.specidx_map
        extrap_maps = self.reference_map[:, np.newaxis] * scaling.T  # Shape: (npix, nfreq)

        # Apply smoothing if requested
        if beam_transfer is not None:
            return smoothed_maps(extrap_maps, beam_transfer)

        return extrap_maps if len(target_freqs) > 1 else extrap_maps.squeeze()

