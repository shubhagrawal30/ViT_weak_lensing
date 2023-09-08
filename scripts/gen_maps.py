import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import h5py, os, sys
from alive_progress import alive_it
from copy import copy
from itertools import repeat

import multiprocessing as mp
num_threads = 200

from pathlib import Path
out_dir = '/pscratch/sd/s/shubh/20230915_224x224_patches/'
Path(out_dir).mkdir(parents=True, exist_ok=True)

instrument = "stage3_forecast"
cosmogrid_dir = "/global/cfs/cdirs/des/cosmogrid/"
fiducial_dir = os.path.join(cosmogrid_dir, instrument, "fiducial")
benchmark_dir = os.path.join(cosmogrid_dir, instrument, "benchmark")
grids_dir = os.path.join(cosmogrid_dir, instrument, "grid")
fil_name = "projected_probes_maps_baryonified512.h5"

def IndexToDeclRa(index, nside,nest= False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)


def gen_from_one_file(fil_name, nside=512, nside_crop=4, \
                      keys=["stage3_lensing" + str(i) for i in range(1, 5)], out_shape=(224, 224, 4)):
    try:
        kappas = {}
        with h5py.File(fil_name) as fil:
            for key in fil['kg'].keys():
                kappas[key] = np.array(fil['kg'][key])

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

        for chan, key in enumerate(keys):
            kappa = np.array(kappas[key])

            for i in range(pairs_.shape[1]):
                m_projected = hp.gnomview(kappa, rot=pairs_[:,i], xsize=xsize,\
                                          no_plot=True,reso=res,return_projected_map=True)
                out_arr[i, :, :, chan] = m_projected
        return out_arr
    except:
        print("failed", fil_name)

def for_one_grid(grid):
    if os.path.exists(os.path.join(out_dir, grid + ".npy")):
        print("already exists", grid)
        return
    try:
        path_grid = os.path.join(grids_dir, grid)
        all_perms_out = np.ndarray((0, 224, 224, 4))
        for perm in os.listdir(path_grid):
            if "perm" not in perm:
                continue
            out_arr = gen_from_one_file(os.path.join(path_grid, perm, fil_name))
            if out_arr is None:
                print("skipping", perm)
            else:
                all_perms_out = np.append(all_perms_out, out_arr, axis=0)
        np.save(os.path.join(out_dir, grid), all_perms_out)
    except:
        print("failed", grid)
    return grid

if __name__ == "__main__":
    grids = os.listdir(grids_dir)
    print(f"found {len(grids)} simulations")
    overwrite = False
    for grid in copy(grids):
        if (not overwrite) and os.path.exists(os.path.join(out_dir, grid + ".npy")):
            grids.remove(grid)
    print(f"running {len(grids)} simulations")
    
    np.random.shuffle(grids)
    
    my_pool = mp.Pool(processes=num_threads)
    for _ in alive_it(my_pool.imap_unordered(for_one_grid, grids), total=len(grids), \
                      bar="checks", spinner="notes"):
        print("done", _)
    my_pool.close()
    my_pool.join()