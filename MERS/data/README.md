# Input maps

These files are not in version control. `MERS/fg_model.py` resolves them
relative to this directory, so they must be placed here by hand before any
of the sky models can be built.

| File | Used by | Source |
|---|---|---|
| `haslam408_dsds_Remazeilles2014.fits` | `SynchrotronExtrapolator` — 408 MHz reference map | Public: destriped/desourced Haslam map, Remazeilles et al. (2014), available from LAMBDA |
| `cnn56arcmin_beta.npy` | `SynchrotronExtrapolator` — default spectral index map (CNN-PL) | Not public: CNN spectral index map at 56 arcmin, from Melis Irfan (added in `3f98f19`) |
| `EM_mean_std.fits` | `FreeFreeExtrapolator` — emission measure | Derived product, see `FreeFreeExtrapolator` |
| `COM_CompMap_freefree-commander_0256_R2.00.fits` | `FreeFreeExtrapolator` — electron temperature (`field=4`) | Public: Planck 2015 Commander free-free map (R2.00), Planck Legacy Archive |
| `gleam_nside512_K_allsky_408MHz.npy` | `ptsrc_powerlaw` — point source foreground at 408 MHz | Not public: GLEAM-derived all-sky map in K (added in `d807e76`) |

`gleam_nside512_K_allsky_50MHz_16freqs.npz` is referenced by a commented-out
line in `fg_model.py` and is not needed by any current code path.

The two non-public maps exist only on the machines of the people who made
them. They were removed from this repository's history on 2026-09-20; the
pre-rewrite history, including those blobs, is kept in
`MERS-history-backup-20260920.bundle`.
