# About

This is the code used to reproduce the results of the paper.

To reproduce the code please install the required packages. We use `uv` for dependencies management.

```bash
uv sync
```

We use `typer` and `uv` to execute script to generate dataset as follows.

```bash
uv run gen_data.py 
```

There will be prompts to ask for parameters of the simulation. We used default values to produce the dataset used in the paper. The scirpt will generate the data in the `data` folder.
For the `paper_000*.ipynb` files, we note the following.

- The `paper_0001_noise_model.ipynb` is used to produce the Figure (1) and (2), related to numerical and analytical study of noise model.  
- The `paper_0002_PSD_SGM_training.ipynb` and `paper_0003_PSD_PGM_training.ipynb` are used to train SGM and PGM predictive models and then we use them to calibrated for the target quantum gate. `model` folder will be created to store optimized model parameters. The control parameters time used will be printed.
- The `paper_0002_PSD_SGM_training.ipynb` and `paper_0003_PSD_PGM_training.ipynb` have to be executed first before the `paper_0004_cc_visualization.ipynb` because we need the trained models to perform analysis. Other than, the times and optimzied values of the control parameter, we load models back to perform prediction and visualize the results.
