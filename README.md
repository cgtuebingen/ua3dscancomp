# Latent Uncertainty-Aware Multi-View SDF Scan Completion
<!--[![Cite this repository](https://img.shields.io/badge/Cite%20this%20repo-BibTeX-blue)](https://crv.pubpub.org/pub/yanc7d1w)-->
[![Watch Video](https://img.shields.io/badge/Watch-Demo-blue)](https://cgtuebingen.github.io/ua3dscancomp)

**Authors**: Faezeh Zakeri, Lukas Ruppert, Raphael Braun, and Hendrik P.A. Lensch

<!--**Conference**: [WACV 2026]()-->

<!--**Arxiv**: [Arxiv 2024]()-->

**Code Repository**: [ua3dscancomp](https://github.com/cgtuebingen/ua3dscancomp)

---

## Abstract

Imperfect reconstructions arising from occlusions, shadows, reflections, and other factors during 3D scanning often result in incomplete sections of the scanned object, with missing parts scattered randomly across its surface. We introduce an uncertainty-aware signed distance field (SDF) latent transformer that leverages uncertainty to identify and reconstruct missing parts based on the global shape of the incomplete scanned object and the immediate neighborhood of the affected regions. To our knowledge, we are the first to utilize uncertainties for SDF shape completion in the latent space. Our model has been trained on the entire Objaverse 1.0 dataset and demonstrates that our uncertainty-aware SDF completion method significantly outperforms previous works both numerically and visually.

---

## Demo Gif
[For high quality video, click here!](https://cgtuebingen.github.io/ua3dscancomp)
<p align="center">
  <img src="docs/demo.gif" alt="Ua3dscancomp Demo">
</p>
<!--![ua3dscancomp Demo](docs/demo.gif)-->

## 📦 Model checkpoint

[Shape Completion on Objaverse](https://huggingface.co/zakeri68/3D-scancomp-objaverse)

[Patchwise Variational Autoencoder (P-VAE) on Shapenet](https://huggingface.co/zakeri68/poc-slt-shapenet-p-vae)


## Dataset
- Evaluation
  - [Objaverse Processed LMDB-Test-Split](https://huggingface.co/datasets/zakeri68/3D-scancomp-objaverse-test-lmdb)

- Train and Validation
  - You can generate train and validation LMDBs using [src/pre_processing](https://github.com/cgtuebingen/ua3dscancomp/tree/main/src/data_preprocessing) scripts provided.

## P-VAE

The source code for P-VAE can be taken from [POC-SLT](https://github.com/cgtuebingen/poc-slt) repository.

## Running Instructions
- Evaluation
  - You can evaluate via [eval_config.py](https://github.com/cgtuebingen/ua3dscancomp/tree/main/src/evaluation/eval_config.py) and given the model checkpoint above.

- Train from scratch
  - You can train from scratch via [train_config.py](https://github.com/cgtuebingen/ua3dscancomp/tree/main/src/training/train_config.py)

## On-the-Fly SDF Calculation
The code for this part will very soon be published in another github repository and will be updated here.

## Project Structure

```bash
├── data/
├── src/
├── docs/
├── requirements.txt
└── README.md
```


<!--## Citation

If you use this work, please cite it as:-->

```bibtex
@article{Zakeri2026ua3dscancomp,
  author  = {Zakeri, Faezeh and Ruppert, Lukas, and Braun, Raphael, and Lensch, Hendrik P.A.},
  title   = {Latent Uncertainty-Aware Multi-View SDF Scan Completion},
  journal = {The IEEE/CVF Winter Conference on Applications of Computer Vision, WACV},
  year    = {2026},
  month   = {March 10},
  note    = {}
}
