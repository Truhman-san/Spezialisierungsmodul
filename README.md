# Synthetic STM Segmentation (Advanced)

This repository contains the code used for the accompanying report of the  
*Spezialisierungsmodul* (JLU Gießen, WS 25/26).

The focus is on training and evaluating U-Net–based multiclass segmentation  
models on synthetically generated STM topographies and analyzing their  
transferability to real STM data.

---

## Repository structure

- src/
- runs/
- plots/
- notebooks/

---

## Typical workflow

### Generate synthetic STM data

```
python src/new_datagen/run.py --out-dir data/synth
```

### Train segmentation model

```
python src/unet_train_fraction.py --data-dir data/synth
```

### Evaluate trained model on synthetic validation data

```
python src/evaluate_unet.py --run-dir runs/<run_name>
```

### Apply trained model to real STM images

```
python src/real_stm_predictions.py --model runs/<run_name>/model
```

---

## Module overview

Short description of the main Python modules contained in the `src` directory.

### Core training and evaluation

- **src/unet_train_fraction.py**  
  Trains a single-output U-Net using a configurable fraction of the synthetic dataset.

- **src/unet_train_2outputs.py**  
  Trains a multi-output U-Net (main segmentation plus optional auxiliary output).

- **src/evaluate_unet.py**  
  Evaluates trained models on synthetic validation data and writes IoU and F1 metrics.

---

### Synthetic data generation

- **src/new_datagen/**  
  Extended synthetic STM data generator with improved background modeling  
  (terraces, gradients, geometric distortions, local contrast variation).

- **src/new_terraces_generator/**  
  Alternative generator focusing on terrace-aware surface construction.

---

### Inference on real STM data

- **src/real_stm_predictions.py**  
  Applies trained models to real STM PNG or TIF images and generates tripanel  
  visualizations.

- **src/ensemble.py**  
  Rule-based combination of multiple trained models at inference time.

- **src/2_step_pipeline.py**  
  Two-stage inference pipeline using structural priors for defect gating.

---

### Semi-supervised and exploratory methods

- **src/ssl/**  
  Experimental implementations of pseudo-labeling and consistency training.

---

## Notes

- Training data are generated synthetically.  
- Real STM data are not included in this repository.  
- Models are evaluated quantitatively only on synthetic data.

---

## Detailed module reference

### src/

- **src/2_step_pipeline.py**  
  Two-stage pipeline (rows head plus defect ensemble); gates defect predictions on background and writes tripanel visualizations for real STM PNG images.

- **src/boost_2dimer.py**  
  Pipeline variant with additional 2-dimer boosting along predicted dimer rows; generates tripanel outputs.

- **src/deleting_masks.py**  
  Deletes segmentation masks without matching images in an external dataset and logs removed paths to `deleted_masks.txt`.

- **src/ensemble.py**  
  Rule-based ensemble of two models with dedicated 2-dimer logic; renders tripanel visualizations from TIF or PNG input.

- **src/ensemble_presentation.py**  
  Presentation-oriented variant of `ensemble.py` using fixed model paths and output directories.

- **src/evaluate_unet.py**  
  Evaluates a single-output U-Net on the test split and writes IoU, F1-score and confusion matrix metrics to JSON and TXT files.

- **src/evaluate_unet_2outputs.py**  
  Evaluation script for multi-output U-Nets (main output plus optional rows head) with separate metrics.

- **src/generate_predictions.py**  
  Loads a trained model, samples validation images and saves prediction overlays and label maps.

- **src/generate_predictions_2outputs.py**  
  Variant of `generate_predictions.py` for multi-output models including optional rows-head predictions.

- **src/plot_learning_curve.py**  
  Reads training statistics from `runs/` and plots learning curves for Mean IoU and Mean F1.

- **src/real_stm_predictions.py**  
  Inference on real STM PNG images with tripanel output (original, grayscale, prediction overlay).

- **src/real_stm_predictions_2dimerweighted.py**  
  Inference on real STM PNG images with heuristic 2-dimer gating prior to visualization.

- **src/real_stm_predictions_2outputs.py**  
  Inference for multi-output models showing defect predictions and optional dimer-row masks.

- **src/train_ssl_consistency.py**  
  Semi-supervised training via consistency loss using weak and strong augmentations with ramp-up and area regularization.

- **src/train_ssl_pseudo.py**  
  Semi-supervised training using an EMA teacher and pseudo-label generation; stores updated model checkpoints.

- **src/unet_train_2outputs.py**  
  Training script for U-Net models with an optional rows head and configurable training fraction.

- **src/unet_train_fraction.py**  
  Training script for single-output U-Net models using configurable subsampling of the synthetic dataset.

- **src/unet_train_fraction_finetuning.py**  
  Finetuning script for existing models using a masked loss function.

- **src/__init__.py**  
  Empty package initializer for `src`.

---

### src/predictions/

- **src/predictions/utils.py**  
  Utility functions for loading PNG images and masks and plotting ground truth versus predictions.

- **src/predictions/__init__.py**  
  Empty package initializer.

---

### src/real_stm/

- **src/real_stm/stm_to_gray.py**  
  Converts raw STM TIF files to 600×600 grayscale PNG images.

- **src/real_stm/__init__.py**  
  Empty package initializer.

---

### src/ssl/

- **src/ssl/consistency.py**  
  Consistency loss functions and STM-specific weak and strong data augmentations.

- **src/ssl/datasets.py**  
  Construction of unlabeled `tf.data` datasets from PNG images.

- **src/ssl/ema.py**  
  Exponential moving average teacher model for pseudo-label training.

- **src/ssl/pseudo_label.py**  
  Pseudo-label generation and masked cross-entropy loss helpers.

- **src/ssl/schedules.py**  
  Linear ramp-up schedules for semi-supervised loss weighting.

- **src/ssl/__init__.py**  
  Empty package initializer.

---

### src/unet/

- **src/unet/config.py**  
  Dataclass definitions for training parameters, dataset configuration and output directories.

- **src/unet/data.py**  
  Dataset builder for images and masks with optional rows head, including augmentation and dataset splits.

- **src/unet/losses.py**  
  Loss functions (cross-entropy, Dice, focal, and masked variants).

- **src/unet/metric_train.py**  
  Mean IoU and Mean F1 metrics for multiclass segmentation.

- **src/unet/model.py**  
  Single-output U-Net architecture with softmax output.

- **src/unet/model_2output.py**  
  U-Net architecture with optional rows head (main plus rows outputs).

- **src/unet/utils.py**  
  Utilities for creating output directories and storing run configurations.

- **src/unet/__init__.py**  
  Empty package initializer.

---

### src/finetuning/

- **src/finetuning/labeling.py**  
  Placeholder file without active logic.

- **src/finetuning/__init__.py**  
  Empty package initializer.

---

### src/new_datagen/

- **src/new_datagen/config.py**  
  Central configuration and base tile definitions for synthetic STM generation.

- **src/new_datagen/generator.py**  
  Generates synthetic STM samples including surface topography and defect masks with optional rotation.

- **src/new_datagen/overlay_engine.py**  
  Placement of random defect signatures on the dimer lattice.

- **src/new_datagen/overlays.py**  
  Mathematical models of defect overlays (single-, double- and 2-dimer signatures).

- **src/new_datagen/run.py**  
  Demo and batch-generation script for synthetic STM samples and visualizations.

- **src/new_datagen/terraces.py**  
  Terrace heightmap generation, dimer rendering and gradient and noise postprocessing.

- **src/new_datagen/tiles.py**  
  Tile repetition and construction of dimer canvases with optional rotation.

- **src/new_datagen/__init__.py**  
  Empty package initializer.

---

### src/new_terraces_generator/

- **src/new_terraces_generator/config.py**  
  Configuration including base tiles and additional row masks for dimer rows.

- **src/new_terraces_generator/dimer_physics.py**  
  Physically modified dimer templates derived from base data.

- **src/new_terraces_generator/generator.py**  
  End-to-end generator for terrace images including rotation, signatures and masks.

- **src/new_terraces_generator/overlay_engine.py**  
  Orientation-aware defect placement with snapping on horizontal and vertical terraces.

- **src/new_terraces_generator/overlays.py**  
  Defect overlays with masked updates restricted to visible regions.

- **src/new_terraces_generator/postprocessing.py**  
  Postprocessing including edge blurring, tilt, drift, zone scaling, artifacts and flow effects.

- **src/new_terraces_generator/run.py**  
  CLI for batch generation of images and masks across parameter variants.

- **src/new_terraces_generator/terraces.py**  
  Terrace layout generation with stepped and irregular edges and orientation handling.

- **src/new_terraces_generator/tiles.py**  
  Tile repetition utilities for arbitrary canvas sizes.

- **src/new_terraces_generator/__init__.py**  
  Empty package initializer.
