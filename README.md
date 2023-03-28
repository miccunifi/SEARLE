# SEARLE
### Zero-shot Composed Image Retrieval With Textual Inversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.15247)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/SEARLE?style=social)](https://github.com/miccunifi/SEARLE)


This is the **official repository** of the [**paper**](https://arxiv.org/abs/2303.15247) "*Zero-**S**hot Compos**E**d Im**A**ge **R**etrieval with Textua**L** Inv**E**rsion*" (**SEARLE**).

>You are currently viewing the code and model repository. If you are looking for more information about the newly-proposed dataset **CIRCO** see the [repository](https://github.com/miccunifi/CIRCO).

## Overview

### Abstract
Composed Image Retrieval (CIR) aims to retrieve a target image based on a query composed of a reference image and a relative caption that describes the difference between the two images. The high effort and cost required for labeling datasets for CIR hamper the widespread usage of existing methods, as they rely on supervised learning. In this work, we propose a new task, Zero-Shot CIR (ZS-CIR), that aims to address CIR without requiring a labeled training dataset. Our approach, named zero-Shot composEd imAge Retrieval with textuaL invErsion (SEARLE), maps the visual features of the reference image into a pseudo-word token in CLIP token embedding space and integrates it with the relative caption. To support research on ZS-CIR, we introduce an open-domain benchmarking dataset named Composed Image Retrieval on Common Objects in context (CIRCO), which is the first dataset for CIR containing multiple ground truths for each query. The experiments show that SEARLE exhibits better performance than the baselines on the two main datasets for CIR tasks, FashionIQ and CIRR, and on the proposed CIRCO.

![](assets/intro.jpg "Workflow of the method")

Workflow of our method. *Top*: in the pre-training phase, we generate pseudo-word tokens of unlabeled images with an optimization-based textual inversion and then distill their knowledge to a textual inversion network. *Bottom*: at inference time on ZS-CIR, we map the reference image to a pseudo-word $S_*$ and concatenate it with the relative caption. Then, we use CLIP text encoder to perform text-to-image retrieval.

## TODO
- [ ] Training and evaluation code
- [ ] Pre-trained models 


## Authors

* [**Alberto Baldrati**](https://scholar.google.com/citations?hl=en&user=I1jaZecAAAAJ)**\***
* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)**\***
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

**\*** Equal contribution. Author ordering was determined by coin flip.

## Citation
```bibtex
@misc{baldrati2023zeroshot,
      title={Zero-Shot Composed Image Retrieval with Textual Inversion}, 
      author={Alberto Baldrati and Lorenzo Agnolucci and Marco Bertini and Alberto Del Bimbo},
      year={2023},
      eprint={2303.15247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.
