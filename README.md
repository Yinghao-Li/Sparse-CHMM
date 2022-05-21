# Sparse-CHMM

This repo contains the code and data of the sparse conditional hidden Markov model (Sparse-CHMM)

## Dependency
This repo is built with Python 3.9.
Please check `./requirements.txt` for required Python packages.
Other package versions are not tested.

## Datasets

We use data provided by the [Wrench benchmark](https://github.com/JieyuZ2/wrench).
The pre-processed datasets are already included in this repo.
You can find them under `./data_constr/<NAME OF DATASET>`.

You can also download the original data from [here](https://drive.google.com/drive/folders/1v55IKG2JN9fMtKJWU48B_5_DcPWGnpTq) (please refer to [this page](https://github.com/JieyuZ2/wrench/blob/main/README.md) for more details), put the unzipped files to the corresponding folders, and use the provided `update_dataset.py` to update data format.

You can find dataset statistics and other information in the `meta.json` files.

Notice that the `lf_f1` section in the `meta.json` files is computed on the training set.

## Run

In the `./label_model` directory You can find several `train-<NAME OF DATASET>.sh` files.
The model parameters presented in the paper are included as default.
You can run the program by going to the directory `cd ./label_model` and running the bash command `./train-<NAME OF DATASET>.sh [GPU ID]`.
The results will be saved in `./label_model/output/<NAME OF DATASET>/` as well as the model checkpoints.
You can also find the log files in `./logs/sparse-chmm-train/`.

You can also run the model with your favorite model parameters by changing the `.sh` files.
The model parameters are defined in `./label_model/sparse_chmm/args.py`.
