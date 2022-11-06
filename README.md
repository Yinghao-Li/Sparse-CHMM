# Sparse Conditional Hidden Markov Model

This repo contains the code and data used in our KDD 2022 paper [Sparse Conditional Hidden Markov Model for Weakly Supervised Named Entity Recognition](https://arxiv.org/abs/2205.14228), which is a follow-up of our previous paper [BERTifying the Hidden Markov Model for Multi-Source Weakly Supervised Named Entity Recognition](https://arxiv.org/abs/2105.12848) published on ACL 2021.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Yinghao-Li/Sparse-CHMM)
![GitHub stars](https://img.shields.io/github/stars/Yinghao-Li/Sparse-CHMM.svg?color=gold)
![GitHub forks](https://img.shields.io/github/forks/Yinghao-Li/Sparse-CHMM?color=9cf)

## 1. Dependency
This repo is built with `Python 3.9`.
Please check `./requirements.txt` for dependent Python packages.
Other package versions are not tested.

## 2. Datasets

We use data provided by the [Wrench benchmark](https://github.com/JieyuZ2/wrench).
The pre-processed datasets are included in this repo.
You can find them under `./data_constr/<NAME OF DATASET>`.

You can also download the original data from [here](https://drive.google.com/drive/folders/1v55IKG2JN9fMtKJWU48B_5_DcPWGnpTq) (please refer to [this page](https://github.com/JieyuZ2/wrench/blob/main/README.md) for more details), put the unzipped files into the corresponding folders, and use the provided `update_dataset.py` to update the data format.

You can find dataset statistics and other information in the `meta.json` files.

Notice that the `lf_f1` section in the `meta.json` files is computed on the training set.

## 3. Model Training

In the `./label_model` directory, you can find several `train-<NAME OF DATASET>.sh` files.
The model parameters presented in the paper are included as default.
You can run the program by going to the directory `cd ./label_model` and running the bash command `./train-<NAME OF DATASET>.sh [GPU ID]`.
The results will be saved in `./label_model/output/<NAME OF DATASET>/` as well as the model checkpoints.
The log files are stored at `./logs/sparse-chmm-train/` by default.

You can also test the model with your favorite model parameters by changing the `.sh` files.
The model parameters are defined in `./label_model/sparse_chmm/args.py`.

## 4. Model Inference

If you would like to use the trained model on new datasets, you can refer to the entry python script `./label_model/sparse-chmm-infer.py` and the bash example `./label_model/infer-laptop.sh`.
Please notice that you should link the argument `test_path` to the new dataset and the argument `output_dir` to the folder containing your trained model.
The program will automatically select the model trained to the latest stage.

The predicted labels are stored in the file `<your output dir>/preds.json`.

## 5. Citation

If you find our work helpful, please consider citing it as
```
@inproceedings{Li-2022-Sparse-CHMM,
    author = {Li, Yinghao and Song, Le and Zhang, Chao},
    title = {Sparse Conditional Hidden Markov Model for Weakly Supervised Named Entity Recognition},
    year = {2022},
    isbn = {9781450393850},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3534678.3539247},
    doi = {10.1145/3534678.3539247},
    booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages = {978-988},
    numpages = {11},
    keywords = {hidden markov model, weak supervision, information extraction, named entity recognition},
    location = {Washington DC, USA},
    series = {KDD '22}
}
```
