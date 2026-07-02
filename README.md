# Know Your Source: A Public Knowledge Store for Media Background Checks

This repository accompanies the paper **"Know Your Source: A Public Knowledge Store for Media Background Checks"** and provides access to the **MEDIAREF** knowledge store.

## 📦 Dataset

The MEDIAREF knowledge store is publicly available on Zenodo:

**https://zenodo.org/records/21136280**

## Overview

Large Language Model (LLM)-based retrieval-augmented generation (RAG) is increasingly used for automated fact-checking and related tasks. While RAG improves transparency by grounding model outputs in retrieved evidence, it often assumes that retrieved sources are reliable. In practice, however, web sources can be conflicting, outdated, or biased.

Recent work on **source-critical reasoning** addresses this challenge through **Media Background Checks (MBCs)**, which evaluate the credibility of evidence sources before they are used for downstream fact verification. Existing approaches for generating MBCs typically depend on proprietary web search APIs, making evaluation costly and difficult to reproduce.

**MEDIAREF** is a publicly available knowledge store of web-sourced documents designed to support reproducible and low-cost evaluation of media background check generation across **200 media sources**. The accompanying paper describes:

- a reproducible methodology for constructing and updating the collection;
- evaluation of widely used LLMs on the MBC generation task;
- automatic and qualitative analyses demonstrating that MEDIAREF enables higher-quality media background checks.

## Repository

Code, evaluation scripts, and annotations will be released soon.

## Citation

If you use MEDIAREF, please cite:

```bibtex
@article{nichols2026knowyoursource,
  title={Know Your Source: A Public Knowledge Store for Media Background Checks},
  author={Nichols, Benjamin and Schlichtkrull, Michael and Ousidhoum, Nedjma},
  journal={arXiv},
  year={2026}
}
```

*(The BibTeX entry will be updated once the paper receives its official arXiv identifier.)*

## License

Please refer to the Zenodo record for dataset licensing and usage terms.
