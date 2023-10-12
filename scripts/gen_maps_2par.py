import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import h5py, os, sys, pickle
from alive_progress import alive_it
from copy import copy
from itertools import repeat

import multiprocessing as mp
num_threads = 100
Z_BINS = 4

from pathlib import Path
out_dir = '/pscratch/sd/s/shubh/20231010_224x224_patches/'
kappa_dir = '/pscratch/sd/s/shubh/20231010_224x224_kappa/'
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(kappa_dir).mkdir(parents=True, exist_ok=True)

params_file = '/pscratch/sd/s/shubh/20231010_224x224_patches/params.csv'
maps_dir = "/global/cfs/cdirs/des/mgatti/darkgrid_SC_nosyst/"

def g2k_sphere(gamma1, gamma2, nside=512, lmax=512*3-1, nosh=True):
    """
    Convert shear to convergence on a sphere. Inputs are all healpix maps.
    """
    KQU_masked_maps = [gamma1, gamma1, gamma2]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!
    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1] * 1.
    almsE[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    E_map = hp.alm2map(almsE, nside=nside, lmax=lmax, pol=False)
    return E_map

def IndexToDeclRa(index, nside, nest=False):
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return - np.degrees(theta - np.pi / 2.), np.degrees(phi)

def get_kappa_allsky(fil_path, nside=512):
    try:
        data = np.load(os.path.join(kappa_dir, fil_path.split('/')[-1].split('.')[0]))
        return data['kappa'], data['Om'], data['sigma8']
    except:
        with open(fil_path, 'rb') as f:
            data = pickle.load(f)[0]
        Om, sigma8 = data['config']['Om'], data['config']['sigma8']
        kappa = np.zeros((Z_BINS, hp.nside2npix(nside)))
        for z in range(Z_BINS):
            gamma1 = data[z+1]['g1']
            gamma2 = data[z+1]['g2']
            kappa[z] = g2k_sphere(gamma1, gamma2, nside=nside)
        np.savez(os.path.join(kappa_dir, fil_path.split('/')[-1].split('.')[0]), kappa=kappa, Om=Om, sigma8=sigma8)
        return kappa, Om, sigma8

def gen_from_one_file(fil_name, nside=512, nside_crop=4, out_shape=(224, 224, 4)):
    fil_path = os.path.join(maps_dir, fil_name)
    if os.path.exists(os.path.join(out_dir, fil_name.split('.')[0] + ".npy")):
        print("already exists", fil_name.split('.')[0])
        return
    try:
        kappas, Om, sigma8 = get_kappa_allsky(fil_path, nside=nside)
        with open(params_file, 'a') as f:
            f.write(f"{fil_name.split('.')[0]},{Om},{sigma8}\n")

        map_indexes = np.arange(hp.nside2npix(nside_crop))
        dec_,ra_ = IndexToDeclRa(map_indexes,nside_crop,nest=False)
        pairs_ = np.vstack([ra_,dec_])
        res = hp.nside2resol(nside, arcmin=True)
        xsize=out_shape[0]

        inds = np.arange(pairs_.shape[1])
        np.random.shuffle(inds)
        pairs_ = pairs_[:, inds]
        # pairs_ = pairs_[:, np.random.choice(np.arange(pairs_.shape[1]), size=10, replace=False)]

        out_arr = np.empty((pairs_.shape[1], *out_shape))

        for chan in range(Z_BINS):
            for i in range(pairs_.shape[1]):
                m_projected = hp.gnomview(kappas[chan], rot=pairs_[:,i], xsize=xsize,\
                                           no_plot=True,reso=res,return_projected_map=True)
                out_arr[i, :, :, chan] = m_projected
        np.save(os.path.join(out_dir, fil_name.split('.')[0]), out_arr)
    except:
        print("failed", fil_name)
    return fil_name

if __name__ == "__main__":
    grids = os.listdir(maps_dir)
    print(f"found {len(grids)} simulations")
    overwrite = False
    for grid in copy(grids):
        if (not overwrite) and os.path.exists(os.path.join(out_dir, grid.split('.')[0] + ".npy")):
            grids.remove(grid)
    print(f"running {len(grids)} simulations")
    
    np.random.shuffle(grids)
    
    my_pool = mp.Pool(processes=num_threads)
    for _ in alive_it(my_pool.imap_unordered(gen_from_one_file, grids), total=len(grids), \
                      bar="checks", spinner="notes"):
        print("done", _)
    my_pool.close()
    my_pool.join()