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
