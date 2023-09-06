"""224x224 tomographic weak lensing kappa simulation maps from CosmoGrid."""

import os
import gc
import healpy as hp
import numpy as np
import pandas as pd
import datasets

_CITATION = """\
"""

_DESCRIPTION = """\
224x224 tomographic weak lensing kappa simulation maps from CosmoGrid.
"""
_HOMEPAGE = ""

_LICENSE = ""

NSIDE = 512
ANG_RES = hp.nside2resol(NSIDE, arcmin=True)
ANG_SIZE = 224 * ANG_RES / 60
NUM_PERMS = 7

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
        if config[0] in DEN_GAL.keys():
            num_patches = np.round(FOOTPRINTS[config[0]] / (ANG_SIZE * ANG_SIZE))
            if len(config) > 1:
                if "half" in config[1]:
                    num_patches //= 2
                if "double" in config[1]:
                    # assert config[0] == "DESY3", "Only DESY3 has double footprint option."
                    num_patches *= 2
                if "onebin" in config[1]:
                    num_patches //= 4
            shape = (224, 224, int(num_patches) * 4)

        features = datasets.Features({   
            "As": datasets.Value("float32"), "bary_Mc": datasets.Value("float32"),
            "bary_nu": datasets.Value("float32"), "H0": datasets.Value("float32"),
            "O_cdm": datasets.Value("float32"), "O_nu": datasets.Value("float32"),
            "Ob": datasets.Value("float32"), "Om": datasets.Value("float32"),
            "ns": datasets.Value("float32"), "s8": datasets.Value("float32"),
            "w0": datasets.Value("float32"),
            "sim_type": datasets.Value("string"), "sim_name": datasets.Value("string"),
            "map": datasets.Array3D(shape=shape, dtype="float32")})

        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, 
            license=_LICENSE, citation=_CITATION,)

    def _split_generators(self, dl_manager):
        data_dir = "/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/data/20230814_224x224"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.txt")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "val.txt")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.txt")
                },
            ),
        ]

    def _generate_examples(self, filepath):
        data_dir = "/pscratch/sd/s/shubh/20230915_224x224_patches/"
        df = pd.read_csv("/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/data/parameters.csv")
        key = 0
        with open(filepath, "r") as reader:
            for filename in reader:
                dat = np.load(os.path.join(data_dir, filename.strip()))
                row = df[df["sim_name"] == filename.strip()[:-4]]
                sim_type, sim_name, As, bary_Mc, bary_nu, H0, O_cdm, O_nu, Ob, Om, ns, s8, w0 = row.values[0]
                config = (self.config.name).split("_")
                if config[0] in DEN_GAL.keys():
                    num_patches_per_key = int(np.round(FOOTPRINTS[config[0]] / (ANG_SIZE * ANG_SIZE)))
                    num_patches_per_perm = dat.shape[0] // NUM_PERMS

                    if len(config) > 1:
                        if "half" in config[1]:
                            num_patches_per_key //= 2
                        if "double" in config[1]:
                            # assert config[0] == "DESY3", "Only DESY3 has double footprint option."
                            num_patches_per_key *= 2
                    
                    num_keys = num_patches_per_perm // num_patches_per_key
                    num_keys = 1

                    if "twice" in config[-1]:
                        # assert config[0] == "DESY3" or config[1] == "half", \
                        #     "Only DESY3 or _half has twice option."
                        num_keys = 2

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
                                "As": As, "bary_Mc": bary_Mc, "bary_nu": bary_nu, "H0": H0,
                                "O_cdm": O_cdm, "O_nu": O_nu, "Ob": Ob, "Om": Om,
                                "ns": ns, "s8": s8, "w0": w0,
                                "sim_type": sim_type, "sim_name": sim_name,
                                "map": img
                            }
                            key += 1
                else:
                    print("Invalid dataset name.")
                    print("Supported: DESY3, LSSTY1, LSSTY10, +_half, +_one_bin, \
                          DESY3_double, +_half_twice, DESY3_twice")

                
