# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""224x224x4 redshift dependent weak lensing kappa simulation maps from CosmoGrid."""


import csv
import os
import gc
import numpy as np
import pandas as pd
import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
224x224x4 redshift dependent weak lensing kappa simulation maps from CosmoGrid.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/data/20230419_224x224"
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """224x224x4 redshift dependent weak lensing kappa simulation maps from CosmoGrid."""

    # DEFAULT_WRITER_BATCH_SIZE = 64

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {   
                    "As": datasets.Value("float32"),
                    "bary_Mc": datasets.Value("float32"),
                    "bary_nu": datasets.Value("float32"),
                    "H0": datasets.Value("float32"),
                    "O_cdm": datasets.Value("float32"),
                    "O_nu": datasets.Value("float32"),
                    "Ob": datasets.Value("float32"),
                    "Om": datasets.Value("float32"),
                    "ns": datasets.Value("float32"),
                    "s8": datasets.Value("float32"),
                    "w0": datasets.Value("float32"),
                    "sim_type": datasets.Value("string"),
                    "sim_name": datasets.Value("string"),
                    "map": datasets.Array3D(shape=(224, 224, 4), dtype="float32")
                    # These are the features of your dataset like images, labels ...
                }
            )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_dir = _URLS[self.config.name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "val.txt"),
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.txt"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        data_dir = _URLS[self.config.name]
        df = pd.read_csv(os.path.join(data_dir, "..", "parameters.csv"))
        key = 0
        with open(filepath, "r") as reader:
            for filename in reader:
                if self.config.name == "first_domain":
                    dat = np.load(os.path.join(data_dir, filename.strip()))
                    row = df[df["sim_name"] == filename.strip()[:-4]]
                    sim_type, sim_name, As, bary_Mc, bary_nu, H0, O_cdm, O_nu, Ob, Om, ns, s8, w0 = row.values[0]
                    # print(row.values[0], d.shape)
                    for d in dat:
                        yield key, {
                            "As": As, "bary_Mc": bary_Mc, "bary_nu": bary_nu, "H0": H0,
                            "O_cdm": O_cdm, "O_nu": O_nu, "Ob": Ob, "Om": Om,
                            "ns": ns, "s8": s8, "w0": w0,
                            "sim_type": sim_type, "sim_name": sim_name,
                            "map": np.array(d)
                        }
                        key += 1
                    del dat
                    gc.collect()
                # else:
                #     yield key, {
                #         "sentence": data["sentence"],
                #         "option2": data["option2"],
                #         "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                #     }