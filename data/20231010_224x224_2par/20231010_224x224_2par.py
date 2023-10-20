"""224x224 tomographic weak lensing kappa simulation maps varying Om and s8."""

import os
import gc
import healpy as hp
import numpy as np
import pandas as pd
import datasets

PATCH_DIR = '/pscratch/sd/s/shubh/20231010_224x224_patches/'
PARAMS_FILE = '/pscratch/sd/s/shubh/20231010_224x224_patches/params.csv'

_CITATION = """\
"""

_DESCRIPTION = """\
224x224 tomographic weak lensing kappa simulation maps varying Om and s8.
"""
_HOMEPAGE = ""

_LICENSE = ""

NSIDE = 512
ANG_RES = hp.nside2resol(NSIDE, arcmin=True)
ANG_SIZE = 224 * ANG_RES / 60
NUM_PERMS = 1

SHAPE_NOISE = 0.26

DEN_GAL = {
    "DESY3": 5.59 * ANG_RES * ANG_RES, # DESY3 Gatti et al. 2020
    "LSSTY1": 11.112 * ANG_RES * ANG_RES, # LSST DESC SRD
    "LSSTY10": 27.737 * ANG_RES * ANG_RES # LSST DESC SRD
}

FOOTPRINTS = {
    "DESY3": 4143, # DESY3 Gatti et al. 2020
    "LSSTY1": 12300, # LSST DESC SRD
    "LSSTY10": 14300 # LSST DESC SRD
}

class NewDataset(datasets.GeneratorBasedBuilder):
    """224x224x4 redshift dependent weak lensing kappa simulation maps from CosmoGrid."""
    VERSION = datasets.Version("1.1.0")
    DEFAULT_WRITER_BATCH_SIZE = 128

    def _info(self):
        config = (self.config.name).split("_")
        num_bins = 4
        if config[0] in DEN_GAL.keys():
            num_patches = np.round(FOOTPRINTS[config[0]] / (ANG_SIZE * ANG_SIZE))
            if len(config) > 1:
                if "half" in config[1]:
                    num_patches //= 2
                if "double" in config[1]:
                    num_patches *= 2
                if "onebin" in config[1]:
                    num_bins = 1    
            shape = (224, 224, int(num_patches) * num_bins)
            

        features = datasets.Features({   
            "Om": datasets.Value("float32"), "s8": datasets.Value("float32"),
            "name": datasets.Value("string"),
            "map": datasets.Array3D(shape=shape, dtype="float32")})

        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, 
            license=_LICENSE, citation=_CITATION,)

    def _split_generators(self, dl_manager):
        file_dir = "/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/data/20231010_224x224_2par/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(file_dir, "train.txt")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(file_dir, "val.txt")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(file_dir, "test.txt")
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(PARAMS_FILE)
        key = 0
        with open(filepath, "r") as reader:
            for filename in reader:
                dat = np.load(os.path.join(PATCH_DIR, filename.strip() + ".npy"))
                row = df[df["name"] == filename.strip()]
                name, Om, s8 = row["name"].values[0], row["Om"].values[0], row["s8"].values[0]
                config = (self.config.name).split("_")
                if config[0] in DEN_GAL.keys():
                    num_patches_per_key = int(np.round(FOOTPRINTS[config[0]] / (ANG_SIZE * ANG_SIZE)))
                    num_patches_per_perm = dat.shape[0] // NUM_PERMS

                    if len(config) > 1:
                        if "half" in config[1]:
                            num_patches_per_key //= 2
                        if "double" in config[1]:
                            num_patches_per_key *= 2
                    
                    num_keys = 1
                    if "twice" in config[-1]:
                        num_keys = 2
                    elif "4times" in config[-1]:
                        num_keys = 4
                    elif "8times" in config[-1]:
                        num_keys = 8
                    elif "16times" in config[-1]:
                        num_keys = 16
                    elif "all" in config[-1]:
                        num_keys = num_patches_per_perm // num_patches_per_key
                    elif "times" in config[-1]:
                        num_keys = int(config[-1].split("times")[0])

                    num_patches_per_key *= 4 # z bins

                    for i in range(NUM_PERMS):
                        d = dat[i*num_patches_per_perm: \
                                (i+1)*num_patches_per_perm].transpose((1, 2, 0, 3)).reshape((224, 224, -1))
                        d += np.random.normal(0, SHAPE_NOISE / np.sqrt(DEN_GAL[config[0]] / 4), d.shape)
                        for j in range(num_keys):
                            img = d[:, :, j*num_patches_per_key: (j+1)*num_patches_per_key]

                            if len(config) > 1 and "onebin" in config[1]:
                                try:
                                    bin_ind = int(config[2])
                                except:
                                    bin_ind = 2 # default to third z-bin
                                img = img[:, :, bin_ind::4]
                            yield key, {
                                "Om": Om, "s8": s8, "name": name, "map": img
                            }
                            key += 1
                else:
                    print("Invalid dataset name.")
                    print("Supported: DESY3, LSSTY1, LSSTY10, +_half, +_one_bin, \
                          +_double, +_half_twice, +_twice")

                
