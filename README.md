# MoCo v2 on Sentinel-2 Imagery

A from-scratch PyTorch implementation of the self-supervised learning method MoCo v2, pre-trained on unlabelled Sentinel-2 satellite images and evaluated on the EuroSAT land-use benchmark.

I built this for the Master's seminar *Exploring Self-Supervised Learning: Models and Practical Implementation* at Technische Hochschule Ingolstadt (summer 2025). The goal was to understand the method by implementing every part of it, on a single GPU in Google Colab.

Everything is in one notebook: [`moco_v2.ipynb`](moco_v2.ipynb).

## What the Notebook Does

1. **Downloads the data.** A subset (up to 5 GB) of the RGB Sentinel-2 images from the [SSL4EO-S12 v1.1](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1) training split.
2. **Loads images on demand.** Each downloaded file is a compressed zarr archive with 64 locations at 4 points in time. The dataset class indexes every single image and reads it from its archive only when it is needed, because extracting everything up front did not fit into memory.
3. **Pre-trains with MoCo v2.** Two augmented views of each image go through a query encoder and a momentum-updated key encoder. A queue of earlier keys supplies the negative examples for the contrastive loss.
4. **Evaluates with a linear probe.** The encoder is frozen and a linear classifier is trained on EuroSAT, a 10-class land-use dataset that is also based on Sentinel-2.

## Setup

| Part | Setting |
|---|---|
| Encoders | Two ResNet-18 networks, trained from scratch (no ImageNet weights) |
| Projection head | 3-layer MLP (100, 50, 25 units) |
| Queue of negatives | 2,048 keys |
| Key encoder momentum | 0.999 |
| Augmentations | Random resized crop, colour jitter, random greyscale, Gaussian blur, horizontal flip |
| Pre-training | 20 epochs, batch size 32, SGD (learning rate 0.001, momentum 0.9, weight decay 1e-6), about 6 hours |
| Linear probe | 20 epochs on EuroSAT, 80/20 train/validation split, SGD (learning rate 0.1) |

## Result

The best validation accuracy of the linear probe on EuroSAT was **59.6 %**.

That is far below the best published models, which reach about 99 % on this benchmark. The gap is expected for this setup: a small ResNet-18, 20 epochs, a 5 GB subset of the pre-training data, a short queue and a batch size of 32. Published MoCo v2 results use a ResNet-50, hundreds of epochs and a queue of 65,536 keys. For comparison, random guessing on 10 classes gives 10 %.

## What I Learned

- **The official implementation assumes several GPUs.** It only supports distributed training and an uncompressed dataset. My first attempt to adapt it ended with a rising loss and 13 % accuracy, close to chance, so I wrote the training loop myself.
- **Data loading was the bottleneck.** Reading single images out of compressed archives is slow, and the memory limit ruled out pre-extracting them. Indexing each image and loading it on demand was the workable compromise.
- **The queue is what makes MoCo practical on small hardware.** The number of negative examples no longer depends on the batch size, which is the main difference to SimCLR.

## Running It

The notebook is written for Google Colab with a GPU runtime and uses `/content` paths. Open it there and run the cells in order. The download cell stops by itself when it reaches the 5 GB limit.

Requirements beyond the Colab defaults: `zarr==2.18.7`.

## Data Sources

- SSL4EO-S12 v1.1: Sentinel-1 and Sentinel-2 imagery for self-supervised learning, hosted by Forschungszentrum Jülich.
- EuroSAT: Helber, Bischke, Dengel, Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* 2017.
- MoCo v2: Chen, Fan, Girshick, He. *Improved Baselines with Momentum Contrastive Learning.* 2020.

This is a study project and is not maintained.
