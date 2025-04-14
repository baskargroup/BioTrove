# [BioTrove: A Large Curated Image Dataset Enabling AI for Biodiversity](https://proceedings.neurips.cc/paper_files/paper/2024/file/b92854f80ba4feefb973959b259dbc2c-Paper-Datasets_and_Benchmarks_Track.pdf)
<!-- Banner links -->
<div style="text-align:center; margin-top: 20px;">
  <a href="https://baskargroup.github.io/BioTrove/" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-Visit-blue" alt="Project Page" style="margin-right:10px;">
  </a>
  <a href="https://huggingface.co/datasets/BGLab/BioTrove" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset%20Card-yellow" alt="Hugging Face Dataset" style="margin-right:10px;">
  </a>
  <a href="https://huggingface.co/BGLab/BioTrove-CLIP" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model%20Card-green" alt="Hugging Face Dataset" style="margin-right:10px;">
  </a>
  <a href="https://pypi.org/project/arbor-process/" target="_blank">
    <img src="https://img.shields.io/badge/PyPI-biotrove--process%200.1.0-orange" alt="PyPI biotrove-process 0.1.0">
  </a>
</div>

<hr style="margin: 10px 0; border: 0; border-top: 0.5px solid #ccc;">

![Banner Image](static/images/Biotrove_top_7_phylum.png)
<hr style="margin: 10px 0; border: 0; border-top: 0.5px solid #ccc;">

### Contents
- [BioTrove: A Large Curated Image Dataset Enabling AI for Biodiversity](#arboretum-a-large-multimodal-dataset-enabling-ai-for-biodiversity)
    - [Contents](#contents)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Validation](#model-validation)
    - [Pre-Run](#pre-run)
    - [Base Command](#base-command)
    - [Baseline Models](#baseline-models)
    - [Existing Benchmarks](#existing-benchmarks)
  - [Acknowledgments](#acknowledgments)
  - [Website License](#website-license)
  - [Citation](#citation)

## Data Preprocessing

**Before using this script, please download the metadata from [Hugging Face](https://huggingface.co/datasets/BGLab/BioTrove-Train)** and pre-process the data using the `biotrove_process` library. The library is located in the `BioTrove-preprocess/biotrove_process` directory. A detailed description can be found in the [README](biotrove-preprocess/README_biotrove_process.md) file.

The library contains scripts to generate machine learning-ready image-text pairs from the downloaded metadata in four steps:

1. *Processing metadata files* to obtain category and species distribution.
2. *Filtering metadata* based on user-defined thresholds and generating shuffled chunks.
3. *Downloading images* based on URLs in the metadata.
4. *Generating text labels* for the images.

## Model Training

We train three models using a modified version of the [BioCLIP/OpenCLIP](https://github.com/Imageomics/bioclip/tree/main/src/training) codebase. Each model is trained for 40 epochs on BioTrove-40M, on 2 nodes, 8xH100 GPUs, on NYU's [Greene](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene) high-performance compute cluster.

We optimize our hyperparameters prior to training with [Ray](https://docs.ray.io/en/latest/index.html). Our standard training parameters are as follows:

```
--dataset-type webdataset 
--pretrained openai 
--text_type random 
--dataset-resampled 
--warmup 5000 
--batch-size 4096 
--accum-freq 1 
--epochs 40
--workers 8 
--model ViT-B-16 
--lr 0.0005 
--wd 0.0004 
--precision bf16 
--beta1 0.98 
--beta2 0.99 
--eps 1.0e-6 
--local-loss 
--gather-with-grad 
--ddp-static-graph 
--grad-checkpointing
```

For more extensive documentation of the training process and the significance of each hyperparameter, we recommend referencing the OpenCLIP and BioCLIP documentation, respectively.


## Model weights

**See the [BioTrove-CLIP](https://huggingface.co/BGLab/BioTrove-CLIP) Model card on HuggingFace to download the trained model checkpoints.**

We released three trained model checkpoints in the [BioTrove-CLIP](https://huggingface.co/BGLab/BioTrove-CLIP) model card on HuggingFace. These CLIP-style models were trained on [BioTrove-Train (40M)](https://huggingface.co/datasets/BGLab/BioTrove-Train) for the following configurations: 

- **BT-CLIP-O:** Trained a ViT-B/16 backbone initialized from the [OpenCLIP's](https://github.com/mlfoundations/open_clip) checkpoint. The training was conducted for 40 epochs.
- **BT-CLIP-B:** Trained a ViT-B/16 backbone initialized from the [BioCLIP's](https://github.com/Imageomics/BioCLIP) checkpoint. The training was conducted for 8 epochs.
- **BT-CLIP-M:** Trained a ViT-L/14 backbone initialized from the [MetaCLIP's](https://github.com/facebookresearch/MetaCLIP) checkpoint. The training was conducted for 12 epochs.


These models were developed for the benefit of the AI community as an open-source product. Thus, we request that any derivative products are also open-source.

## Model Validation

For validating the zero-shot accuracy of our trained models and comparing to other benchmarks, we use the [VLHub](https://github.com/penfever/vlhub) repository with some slight modifications.

### Pre-Run

After cloning this repository and navigating to the `BioTrove/model_validation` directory, we recommend installing all the project requirements into a conda container; `pip install -r requirements.txt`. Also, before executing a command in VLHub, please add `BioTrove/model_validation/src` to your PYTHONPATH.

```bash
export PYTHONPATH="$PYTHONPATH:$PWD/src";
```

### Base Command

A basic `BioTrove` model evaluation command can be launched as follows. This example would evaluate a CLIP-ResNet50 checkpoint whose weights resided at the path designated via the `--resume` flag on the ImageNet validation set, and would report the results to Weights and Biases.

```bash
python src/training/main.py --batch-size=32 --workers=8 --imagenet-val "/imagenet/val/" --model="resnet50" --zeroshot-frequency=1 --image-size=224 --resume "/PATH/TO/WEIGHTS.pth" --report-to wandb
```

### Baseline Models

We compare our trained checkpoints to three strong baselines. We describe our baselines in the table below, including the required flags to evaluate them.


| Model Name  | Origin                                       | Path to checkpoint                        | Runtime Flags                                           |
|-------------|----------------------------------------------|-------------------------------------------|---------------------------------------------------------|
| BioCLIP     | https://arxiv.org/abs/2311.18803             | https://huggingface.co/imageomics/bioclip | --model ViT-B-16 --resume "/PATH/TO/bioclip_ckpt.bin"   |
| OpenAI CLIP | https://arxiv.org/abs/2103.00020             | Downloads automatically                   | --model ViT-B-16 --pretrained=openai                    |
| MetaCLIP-cc | https://github.com/facebookresearch/MetaCLIP | Downloads automatically                   | --model ViT-L-14-quickgelu --pretrained=metaclip_fullcc |


### Existing Benchmarks

In the BioTrove paper, we report results on the following established benchmarks from prior scientific literature: [Birds525](https://www.kaggle.com/datasets/gpiosenka/100-bird-species), [BioCLIP-Rare](https://huggingface.co/datasets/imageomics/rare-species), [IP102 Insects](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset), [Fungi](http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20M-images.tar.gz), [Deepweeds](https://www.kaggle.com/datasets/imsparsh/deepweeds), and [Confounding Species](https://arxiv.org/abs/2306.02507). We also introduce three new benchmarks: BioTrove-Balanced, BioTrove-LifeStages, and BioTrove-Unseen.

Our package expects a valid path to each image to exist in its corresponding metadata file; therefore, **metadata CSV paths must be updated before running each benchmark.**

| Benchmark Name      | Images URL                                                             | Metadata Path                                       | Runtime Flag(s)                     |
|---------------------|------------------------------------------------------------------------|-----------------------------------------------------|-------------------------------------|
| BioTrove-Balanced  |        https://huggingface.co/datasets/BGLab/BioTrove-Train        | https://huggingface.co/datasets/BGLab/BioTrove/tree/main/BioTrove-benchmark/BioTrove-Balanced.csv | --arbor-val --taxon MY_TAXON      |
| BioTrove-Lifestages | https://huggingface.co/datasets/BGLab/BioTrove-Train             | https://huggingface.co/datasets/BGLab/BioTrove/tree/main/BioTrove-benchmark/BioTrove-LifeStages.csv | --lifestages --taxon MY_TAXON   |
| BioTrove-Unseen      | https://huggingface.co/datasets/BGLab/BioTrove-Train             | https://huggingface.co/datasets/BGLab/BioTrove/tree/main/BioTrove-benchmark/BioTrove-Unseen.csv | --arbor-rare --taxon MY_TAXON |
| BioCLIP Rare        | https://huggingface.co/datasets/imageomics/rare-species                | model_validation/metadata/bioclip-rare-metadata.csv | --bioclip-rare --taxon MY_TAXON     |
| Birds525            | https://www.kaggle.com/datasets/gpiosenka/100-bird-species             | model_validation/metadata/birds525_metadata.csv     | --birds /birds525 --ds-filter birds |
| Confounding Species | TBD                                                                    | model_validation/metadata/confounding_species.csv   | --confounding                       |
| Deepweeds           | https://www.kaggle.com/datasets/imsparsh/deepweeds                     | model_validation/metadata/deepweeds_metadata.csv    | --deepweeds                         |
| Fungi               | http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20M-images.tar.gz | model_validation/metadata/fungi_metadata.csv        | --fungi                             |
| IP102 Insects       | https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset                  | model_validation/metadata/ins2_metadata.csv         | --insects2                          |

## Acknowledgments

If you find this repository useful, please consider citing these related papers --

VLHub

```bibtex
@article{
  feuer2023distributionally,
  title={Distributionally Robust Classification on a Data Budget},
  author={Benjamin Feuer and Ameya Joshi and Minh Pham and Chinmay Hegde},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=D5Z2E8CNsD},
  note={}
}
```

BioCLIP

```bibtex
@INPROCEEDINGS{Stevens2024-hc,
  title     = "{BioCLIP}: A Vision Foundation Model for the Tree of Life",
  author    = "Stevens, Samuel and Wu, Jiaman and Thompson, Matthew J and
               Campolongo, Elizabeth G and Song, Chan Hee and Carlyn, David
               Edward and Dong, Li and Dahdul, Wasila M and Stewart, Charles and
               Berger-Wolf, Tanya and Chao, Wei-Lun and Su, Yu",
  booktitle = "Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)",
  pages     = "19412--19424",
  year      =  2024,
  url       = "https://openaccess.thecvf.com/content/CVPR2024/papers/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.pdf"
}
```

OpenCLIP

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.


## Citation
```bibtex
@misc{yang2024arboretumlargemultimodaldataset,
        title={Arboretum: A Large Multimodal Dataset Enabling AI for Biodiversity}, 
        author={Chih-Hsuan Yang, Benjamin Feuer, Zaki Jubery, Zi K. Deng, Andre Nakkab, Md Zahid Hasan, Shivani Chiranjeevi, Kelly Marshall, Nirmal Baishnab, Asheesh K Singh, Arti Singh, Soumik Sarkar, Nirav Merchant, Chinmay Hegde, Baskar Ganapathysubramanian},
        year={2024},
        eprint={2406.17720},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2406.17720}, 
  }
```
