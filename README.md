This branch contains the repro-pack for the paper at https://github.com/barbagroup/jcs_paper_pinn

## Description of the repro-pack

All cases were done with Singularity/Apptainer images.
The definition files of the images are in `resources/singularityfiles`.
Two images were included here: one for PetIBM, and the other one for NVIDIA's Modulus toolkit.
The PetIBM image was used for cases in the `petibm` folder, while the Modulus image was used
by cases in the `modulus` folder.

## To use pre-generated data for plotting

The figures in the paper can be re-plotted using pre-generated data.
In other words, it's not necessary to re-run all cases to generate data.

First, if you have not done so, clone this repository, which contains only the scripts to run cases and post-processing:
```
$ git clone \
    --branch jcs-paper \
    https://github.com/barbagroup/chuang-dissertation-repro-pack.git \
    paper-repro-pack
```
Note that we must use `jcs-paper` branch as this branch is specifically for this paper.
We also renamed the local repo to `paper-repro-pack`.
However, the renaming is just for our convenience.

Download the data tarbal accroding to the reproducibility statement in the paper.
And extract the tarbal content into the repro pack (e.g., `paper-repro-pack`) with
```
$ tar -xf <data tarbal name> -C paper-repro-pack --strip 1
```
Then you should see folders like `outputs`, `output`, or `log` in each case folder, meaning data
were extracted to the corresponding folders.

Finally, follow the section of creating figures in this README to re-plot the figures.


## To generate data from scratch

### To run cases under the `petibm` folder:

For those under `petibm/tgv-2d-re100`, use the following command in each case folder:
```
CUDA_VISIBLE_DEVICES=<list of cuda devices> mpiexec \
    -n <number of MPI ranks> \
    apptainer exec --nv \
        <path to PetIBM' singularity/apptainer image> \
        petibm-navierstokes
```

For the two cylinder flows (`re200` and `re40`), use the following command in each case folder:
```
CUDA_VISIBLE_DEVICES=<list of cuda devices> mpiexec \
    -n <number of MPI ranks> \
    apptainer exec --nv \
        <path to PetIBM's singularity/apptainer image> \
        petibm-decoupledibpm
```

If there several variants of `config.yaml` under a PetIBM case folder, then append
`-config <path to the variant>`.
For example, there's a `config.yaml.refined` in the `re200` case, then use
`-config ./config.yaml.refined` if currently under the case folder.

### To run cases under the `modulus` folder:

Each case has a `main.py`.
Just use the python interpreter in the Modulus image to run a case (if currently under the case
folder):
```
CUDA_VISIBLE_DEVICES=<the id of the GPU> apptainer exec \
    <path to Modulus' singularity/apptainer> python ./main.py
```

## Creating figures in the paper

* Figure 2:
  Execute `petibm/tgv-2d-re100/postprocessing/create_convergence_for_vv.py`, and the figure will be
  saved to `petibm/figures/petibm-tgv-2d-re100-convergence.png`

* Figure 3, 4, and 5:
  1. Execute `modulus/tgv-2d-re100/postprocessing/gather_data.py` to post-process raw data
  2. Execute `modulus/tgv-2d-re100/postprocessing/create_cyclic_swa_test_plots.py`
  3. Execute `modulus/tgv-2d-re100/postprocessing/create_base_case_contours.py`
  4. The three figures will be saved to
     a. `modulus/tgv-2d-re100/figures/cyclic-swa-tests/learning-rate-hist.png`
     b. `modulus/tgv-2d-re100/figures/cyclic-swa-tests/nl3-nn128-npts8192.png`
     b. `modulus/tgv-2d-re100/figures/contours/nl3-nn256-npts4096-t40.0.png`

* Figure 6 to 9 and table 1:
  1. Execute `modulus/cylinder-2d-re40/postprocessing/gather_data.py` to post-process raw data
  2. Execute `modulus/cylinder-2d-re40/postprocessing/create_training_hist.py`
  3. Execute `modulus/cylinder-2d-re40/postprocessing/create_force_hist.py`
  4. Execute `modulus/cylinder-2d-re40/postprocessing/create_surface_p_plots.py`
  5. Execute `modulus/cylinder-2d-re40/postprocessing/create_contours.py`
  6. Execute `modulus/cylinder-2d-re40/postprocessing/create_table.py`
  7. The figures will be saved to:
     a. `modulus/cylinder-2d-re40/figures/loss-hist.png`
     b. `modulus/cylinder-2d-re40/figures/drag-lift-coeffs.png`
     c. `modulus/cylinder-2d-re40/figures/surface-pressure.png`
     d. `modulus/cylinder-2d-re40/figures/contour-comparison.png`
  8. The TeX expression of the table will be saved to `modulus/cylinder-2d-re40/tables/drag-lift-coeff.tex`

* Figure 10 - 16:
  1. Execute `modulus/cylinder-2d-re200/postprocessing/gather_data.py` to post-process raw data
     (The peak memory usage may be about 10GB.)
  2. Execute `modulus/cylinder-2d-re200/postprocessing/create_training_hist.py`
  3. Execute `modulus/cylinder-2d-re200/postprocessing/create_training_hist.py`
  4. Execute `modulus/cylinder-2d-re200/postprocessing/create_contours.py`
  7. The figures will be saved to:
     a. `modulus/cylinder-2d-re200/figures/loss-hist.png`
     b. `modulus/cylinder-2d-re200/figures/drag-lift-coeffs.png`
     c. `modulus/cylinder-2d-re200/figures/contour-comparison-u.png`
     d. `modulus/cylinder-2d-re200/figures/contour-comparison-v.png`
     e. `modulus/cylinder-2d-re200/figures/contour-comparison-p.png`
     f. `modulus/cylinder-2d-re200/figures/contour-comparison-omega_z.png`
     g. `modulus/cylinder-2d-re200/figures/contour-comparison-steady.png`

* Figure 17 - 18:
  1. Execute `modulus/cylinder-2d-re200/postprocessing/gather_refined_data.py` to post-process raw data
  2. Execute `modulus/cylinder-2d-re200/postprocessing/plot_refined_contourf.py`
  7. The figures will be saved to:
     a. `modulus/cylinder-2d-re200/figures/refined/vorticity_z.png`
     b. `modulus/cylinder-2d-re200/figures/refined/qcriterion.png`

* Figure 19, 20, and those in the appendix:
  1. Execute `modulus/cylinder-2d-re200/postprocessing/koopman_decomp.py` to post-process raw data
  2. Execute `modulus/cylinder-2d-re200/postprocessing/plot_koopman.py`
  7. The figures will be saved to:
     a. `modulus/cylinder-2d-re200/figures/koopman/koopman_eigenvalues_complex.png`
     b. `modulus/cylinder-2d-re200/figures/koopman/koopman_mode_strength.png`
     c. all other figures in `modulus/cylinder-2d-re200/figures/koopman`
