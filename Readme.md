# Cytobench
<img src="https://github.com/lazzaronico/cytobench/blob/main/logo.png" alt="Cytobench Logo" width="400"  align="left" style="margin-right: 10px;" />

Welcome to Cytobench, the friendliest benchmarking library in the village!

This repository is meant as a demo implementation of a scoring pipeline to evaluate generative models capabilities of recapitulating target empirical distributions via Pointwise Empirical Distance estimation, as presented in (paper_link). Most of the functionalities are presented in the Jupyter notebooks, which can be run interactively via Colab without having to create a local enviornment.

The code is not meant as a performant implementation, nor has it been protected by proper safety checks: do not use this in production environments!

You can install the latest version via pip (requires python 3.10):
```bash
pip install cytobench
```

If you use this library in your work please cite: (citation)