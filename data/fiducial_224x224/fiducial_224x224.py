"""224x224 tomographic weak lensing kappa maps at fiducial cosmology"""

import os
import gc
import healpy as hp
import numpy as np
import pandas as pd
import datasets

_CITATION = """\
"""

data_dir = "/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/data/fiducial_224x224/"

_DESCRIPTION = """\
224x224 tomographic weak lensing kappa maps at fiducial cosmology
"""
_HOMEPAGE = ""

_LICENSE = ""

NSIDE = 512
ANG_RES = hp.nside2resol(NSIDE, arcmin=True)
ANG_SIZE = 224 * ANG_RES / 60
NUM_PERMS = 1

FIDUCIAL_COSMO = {"sim_type": "fiducial", "sim_name": "cosmo_fiducial",
                  "As": 3.058900589392764e-09, "bary_Mc": 66000000000000.0, "bary_nu": 0.0,
                  "H0": 67.36, "O_cdm": 0.209277442262, "O_nu": 0.0014225577379999993, 
                  "Ob": 0.0493, "Om": 0.26, "ns": 0.9649, "s8": 0.84, "w0": -1.0}

SHAPE_NOISE = 0.26

DEN_GAL = {
    "DESY3": 5.59, # DESY3 Gatti et al. 2020
    "LSSTY1": 11.112, # LSST DESC SRD
    "LSSTY10": 27.737 # LSST DESC SRD
}

FOOTPRINTS = {
    "DESY3": 4143, # DESY3 Gatti et al. 2020
    "LSSTY1": 12300, # LSST DESC SRD
    "LSSTY10": 14300 # LSST DESC SRD
}

class NewDataset(datasets.GeneratorBasedBuilder):
    """224x224 tomographic weak lensing kappa maps at fiducial cosmology"""
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
                    assert config[0] == "DESY3", "Only DESY3 has double footprint option."
                    num_patches *= 2
                if "onebin" in config[1]:
                    num_patches //= 4
            shape = (224, 224, int(num_patches) * 4)
        elif config[0] == "old":
            if config[1] == "DES":
                shape = (224, 224, 40)
            elif config[1] == "DEShalf":
                shape = (224, 224, 20)
            elif config[1] == "DESonebin":
                shape = (224, 224, 10)
            

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
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                },
            )
        ]

    def _generate_examples(self):
        key = 0
        filenames = os.listdir(data_dir)
        for filename in filenames:
            if "perm" not in filename:
                continue
            dat = np.load(os.path.join(data_dir, filename.strip()))
            sim_type, sim_name, As, bary_Mc, bary_nu, H0, O_cdm, O_nu, Ob, Om, ns, s8, w0 = FIDUCIAL_COSMO.values()
            config = (self.config.name).split("_")
            if config[0] in DEN_GAL.keys():
                num_patches_per_key = int(np.round(FOOTPRINTS[config[0]] / (ANG_SIZE * ANG_SIZE)))
                num_patches_per_perm = dat.shape[0] // NUM_PERMS

                if len(config) > 1:
                    if "half" in config[1]:
                        num_patches_per_key //= 2
                    if "double" in config[1]:
                        assert config[0] == "DESY3", "Only DESY3 has double footprint option."
                        num_patches_per_key *= 2
                
                num_keys = num_patches_per_perm // num_patches_per_key
                num_keys = 1

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
            elif config[0] == "old":
                for i in range(NUM_PERMS):
                    num_patches_per_perm = 10
                    d = dat[i*num_patches_per_perm: \
                            (i+1)*num_patches_per_perm].transpose((1, 2, 0, 3)).reshape((224, 224, -1))
                    if config[1] == "DEShalf":
                        d = d[:, :, :20]
                    elif config[1] == "DESonebin":
                        d = d[:, :, 2::4]
                    d += np.random.normal(0, SHAPE_NOISE / np.sqrt(DEN_GAL["DESY3"] / 4), d.shape)
                    yield key, {
                        "As": As, "bary_Mc": bary_Mc, "bary_nu": bary_nu, "H0": H0,
                        "O_cdm": O_cdm, "O_nu": O_nu, "Ob": Ob, "Om": Om,
                        "ns": ns, "s8": s8, "w0": w0,
                        "sim_type": sim_type, "sim_name": sim_name,
                        "map": d
                    }
                    key += 1
            else:
                print("Invalid dataset name.")
                print("Supported: DESY3, LSSTY1, LSSTY10, +_half, +_one_bin, \
                        DESY3_double, old_DES, old_DEShalf, old_DESonebin")

            
