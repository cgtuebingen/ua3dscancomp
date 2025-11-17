# Latent Uncertainty-Aware Multi-View SDF Scan Completion
[![Cite this repository](https://img.shields.io/badge/Cite%20this%20repo-BibTeX-blue)](https://crv.pubpub.org/pub/yanc7d1w)
[![Watch Video](https://img.shields.io/badge/Watch-Demo-blue)](https://cgtuebingen.github.io/poc-slt/)

**Authors**: Faezeh Zakeri, Lukas Ruppert, Raphael Braun, and Hendrik P.A. Lensch

<!--**Conference**: [WACV 2026]()-->

<!--**Arxiv**: [Arxiv 2024]()-->

**Code Repository**: [ua3dscancomp](https://github.com/cgtuebingen/ua3dscancomp)

---

## Project Progress Checklist

- [ ] Upload pretrained models
- [ ] Upload train and validation splits
- [ ] Release main codebase
- [ ] Add evaluation scripts
- [ ] Clean up and document configs
- [ ] Add license info
<!--- [ ] Upload Dataset LMDB-->
---

## Abstract

Imperfect reconstructions arising from occlusions, shadows, reflections, and other factors during 3D scanning often result in incomplete sections of the scanned object, with missing parts scattered randomly across its surface. We introduce an uncertainty-aware signed distance field (SDF) latent transformer that leverages uncertainty to identify and reconstruct missing parts based on the global shape of the incomplete scanned object and the immediate neighborhood of the affected regions. To our knowledge, we are the first to utilize uncertainties for SDF shape completion in the latent space. Our model has been trained on the entire Objaverse 1.0 dataset and demonstrates that our uncertainty-aware SDF completion method significantly outperforms previous works both numerically and visually.

---

## Demo Gif
<!--[For high quality video, click here!](https://cgtuebingen.github.io/poc-slt/)
<p align="center">
  <img src="docs/demo.gif" alt="POC-SLT Demo">
</p>
<!--![POC-SLT Demo](docs/demo.gif)-->-->

## Project Structure

```bash
├── data/
├── src/
├── results/
├── requirements.txt
├── train.py
├── eval.py
└── README.md
```


<!--## Citation

If you use this work, please cite it as:-->

<!--```bibtex
@article{Zakeri2025POC,
  author  = {Zakeri, Faezeh and Braun, Raphael and Ruppert, Lukas and Lensch, Hendrik P.A.},
  title   = {POC-SLT: Partial Object Completion with SDF Latent Transformers},
  journal = {Proceedings of the Conference on Robots and Vision},
  year    = {2025},
  month   = {May 27},
  note    = {https://crv.pubpub.org/pub/yanc7d1w}
}
You can also find citation metadata files or by clicking **Cite this repository** on the right.-->
