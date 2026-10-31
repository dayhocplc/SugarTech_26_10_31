# Supplementary file – for online publication only



Supplementary material for: *Detection completeness limits deep learning weed detection in field-grown sugarcane*.



These tables give in full the sensitivity analyses summarised in the main text. They are referred to there as Supplementary Table S1 to S6. The main text stands alone without them.



**Supplementary Table S1. Training and inference configuration of the three bounding-box models, in full. All were trained for 50 epochs with AdamW, an initial learning rate of 0.001, cosine annealing, weight decay 0.0001, a three-epoch linear warm-up and an effective batch of 16 realised by gradient accumulation. Predictions scoring below 0.05 were discarded before any metric was computed. A seed fixes the Python, NumPy and PyTorch generators, and therefore the initialisation of the detection heads, the data ordering and the sampling of the training-time horizontal flip; the data partition and the augmented images are identical across seeds.**

| Model | Framework | Input size | Initialisation | Max detections per image | NMS IoU threshold | Checkpoint evaluated |
|---|---|---|---|---|---|---|
| RetinaNet | Detectron2 v0.6 | 1024 × 1024 | COCO-pretrained | 100 | 0.50 | final epoch |
| Faster R-CNN | Detectron2 v0.6 | 1024 × 1024 | COCO-pretrained | 100 | 0.50 (head), 0.70 (RPN) | final epoch |
| YOLOv10 | Ultralytics v8.3 | 640 × 640 | COCO-pretrained | 300 | none (end-to-end head) | final epoch |


**Supplementary Table S2. Weed-class Recall, Precision and F1-score in the full variant of the confusion matrix, as a function of the confidence threshold, at a fixed IoU threshold of 0.50. Values are means over each model's converged training runs. Confidence scores are not calibrated to a common scale between frameworks, so a single threshold does not represent the same operating point for every model.**

| Confidence threshold | RetinaNet (R / P / F1) | Faster R-CNN (R / P / F1) | YOLOv10 (R / P / F1) |
|---|---|---|---|
| 0.05 | 0.603 / 0.129 / 0.213 | 0.524 / 0.397 / 0.452 | 0.461 / 0.482 / 0.471 |
| 0.10 | 0.569 / 0.243 / 0.341 | 0.508 / 0.440 / 0.471 | 0.437 / 0.525 / 0.477 |
| 0.20 | 0.522 / 0.369 / 0.432 | 0.490 / 0.478 / 0.484 | 0.408 / 0.564 / 0.474 |
| 0.30 | 0.501 / 0.436 / 0.466 | 0.480 / 0.498 / 0.489 | 0.388 / 0.591 / 0.468 |
| 0.40 | 0.476 / 0.478 / 0.477 | 0.470 / 0.514 / 0.491 | 0.373 / 0.614 / 0.464 |
| 0.50 | 0.456 / 0.514 / 0.483 | 0.463 / 0.529 / 0.493 | 0.357 / 0.634 / 0.457 |
| 0.60 | 0.436 / 0.546 / 0.485 | 0.455 / 0.541 / 0.494 | 0.340 / 0.649 / 0.446 |
| 0.70 | 0.417 / 0.577 / 0.484 | 0.444 / 0.552 / 0.492 | 0.321 / 0.672 / 0.434 |
| 0.80 | 0.396 / 0.616 / 0.482 | 0.432 / 0.566 / 0.490 | 0.293 / 0.697 / 0.412 |
| 0.90 | 0.355 / 0.665 / 0.462 | 0.415 / 0.596 / 0.489 | 0.236 / 0.737 / 0.357 |


**Supplementary Table S3. The same metrics as a function of the IoU threshold at which a prediction is accepted as matching an annotation, at a fixed confidence threshold of 0.30. The difference between the rows measures how much of the apparent shortfall in Recall is imprecise localisation rather than failure to detect.**

| IoU threshold | RetinaNet (R / P / F1) | Faster R-CNN (R / P / F1) | YOLOv10 (R / P / F1) |
|---|---|---|---|
| 0.10 | 0.632 / 0.550 / 0.588 | 0.614 / 0.637 / 0.625 | 0.481 / 0.733 / 0.580 |
| 0.30 | 0.600 / 0.522 / 0.558 | 0.582 / 0.604 / 0.593 | 0.457 / 0.696 / 0.551 |
| 0.50 | 0.501 / 0.436 / 0.466 | 0.480 / 0.498 / 0.489 | 0.388 / 0.591 / 0.468 |


**Supplementary Table S4. Weed-class Recall in the full variant, stratified by the area of the annotated bounding box, at a confidence threshold of 0.30 and an IoU threshold of 0.50. The upper block uses the absolute size bands of the COCO protocol; the lower block uses quartiles of the area distribution of this test partition, which divide the data evenly and are therefore the more reliable instrument here. Areas are measured on the 1280 px exported frame.**

| Size band | Weed instances | RetinaNet | Faster R-CNN | YOLOv10 |
|---|---|---|---|---|
| COCO small (< 32 × 32 px) | 42 | 0.060 | 0.057 | 0.029 |
| COCO medium (32–96 px) | 185 | 0.280 | 0.280 | 0.227 |
| COCO large (> 96 × 96 px) | 667 | 0.606 | 0.573 | 0.461 |
| Q1 (< 9,092 px²) | 224 | 0.237 | 0.233 | 0.188 |
| Q2 (9,092–22,606 px²) | 223 | 0.489 | 0.474 | 0.396 |
| Q3 (22,606–53,437 px²) | 223 | 0.572 | 0.549 | 0.454 |
| Q4 (> 53,437 px²) | 224 | 0.754 | 0.696 | 0.533 |


**Supplementary Table S5. Average precision and average recall under the COCO protocol, by size band, averaged over each model's converged training runs. The ground-truth test partition contains 43 small, 215 medium and 1061 large instances across both classes, so the small-band figures rest on few objects and are reported for the trend rather than as values.**

| Metric | RetinaNet | Faster R-CNN | YOLOv10 |
|---|---|---|---|
| AP | 28.5 | 26.7 | 27.5 |
| AP50 | 57.0 | 53.6 | 49.8 |
| AP75 | 24.7 | 24.1 | 26.3 |
| AP small | 0.8 | 0.9 | 0.9 |
| AP medium | 7.9 | 9.1 | 8.4 |
| AP large | 31.6 | 29.4 | 30.2 |
| AR@100 | 40.3 | 36.3 | 36.9 |
| AR small | 4.9 | 1.7 | 1.4 |
| AR medium | 20.1 | 17.3 | 15.7 |
| AR large | 43.6 | 39.6 | 40.3 |


**Supplementary Table S6. Sensitivity of YOLOv10 to the choice of evaluated checkpoint. Detectron2 writes only a final checkpoint, so the two Detectron2 models admit no equivalent choice; taking each framework's default would therefore have given YOLOv10 an advantage of 2.5 AP50 and inflated its between-seed standard deviation by a factor of 6. The final-epoch rule is used throughout the main text.**

| Checkpoint rule | AP50 (mean ± SD across seeds) | Runs |
|---|---|---|
| Final epoch (protocol used here) | 49.83 ± 0.60 | 5 |
| Best epoch on the validation partition | 52.32 ± 3.73 | 5 |


**Supplementary Table S7. Welch's t-test on the seed-level values of each metric, for all three pairwise comparisons, with P-values adjusted across the three comparisons by the Holm–Bonferroni procedure. These tests hold the test partition fixed and therefore reflect training variability only; the two-level bootstrap reported in the main text releases both sources at once and is the basis for the paper's statements about whether architectures differ.**

| Metric | Comparison | Difference | P (raw) | P (Holm) | Hedges' g [95% CI] |
|---|---|---|---|---|---|
| AP50 | Faster R-CNN − RetinaNet | -3.443 | 0.005 | 0.005 | -2.84 [-5.02, -0.65] |
| AP50 | Faster R-CNN − YOLOv10 | +3.773 | 0.002 | 0.004 | +3.12 [+0.97, +5.27] |
| AP50 | RetinaNet − YOLOv10 | +7.216 | 0.000 | 0.000 | +13.73 [+5.17, +22.29] |
| Weed Recall (full) | Faster R-CNN − RetinaNet | -0.021 | 0.042 | 0.042 | -1.66 [-3.32, +0.01] |
| Weed Recall (full) | Faster R-CNN − YOLOv10 | +0.092 | 0.000 | 0.000 | +7.54 [+3.12, +11.96] |
| Weed Recall (full) | RetinaNet − YOLOv10 | +0.113 | 0.000 | 0.000 | +7.87 [+2.85, +12.89] |
| Weed F1 (full) | Faster R-CNN − RetinaNet | +0.022 | 0.005 | 0.016 | +2.47 [+0.46, +4.48] |
| Weed F1 (full) | Faster R-CNN − YOLOv10 | +0.021 | 0.006 | 0.016 | +2.12 [+0.40, +3.84] |
| Weed F1 (full) | RetinaNet − YOLOv10 | -0.002 | 0.761 | 0.761 | -0.19 [-1.51, +1.13] |


**Supplementary Table S8. Two-level bootstrap over 1000 iterations, resampling the 124 test images and the converged training runs together and taking each difference within an iteration. This is the interval the main text refers to wherever it states that two architectures do or do not differ.**

| Metric | Comparison | Difference | 95% interval | Excludes zero |
|---|---|---|---|---|
| Weed Recall | Faster R-CNN − RetinaNet | -0.020 | [-0.060, +0.018] | no |
| Weed F1-score | Faster R-CNN − RetinaNet | +0.023 | [-0.011, +0.056] | no |
| Weed Recall | Faster R-CNN − YOLOv10 | +0.094 | [+0.046, +0.144] | yes |
| Weed F1-score | Faster R-CNN − YOLOv10 | +0.021 | [-0.020, +0.063] | no |
| Weed Recall | RetinaNet − YOLOv10 | +0.114 | [+0.069, +0.164] | yes |
| Weed F1-score | RetinaNet − YOLOv10 | -0.002 | [-0.040, +0.037] | no |


**Supplementary Table S9. Weed-class Recall, Precision and F1-score in the full variant, computed separately on each capture session of the test partition, with the number of source images each session contributed to the three partitions. Values are means over each model's converged runs. The sessions differ simultaneously in plot, crop age and illumination, so the spread between them cannot be attributed to any single factor. Note that the random partitioning left the validation partition with no images from the first session.**

| Capture session | Source images (train / valid / test) | RetinaNet (R / P / F1) | Faster R-CNN (R / P / F1) | YOLOv10 (R / P / F1) |
|---|---|---|---|---|
| 09h (40-day plot) | 335 / 0 / 28 | 0.398 / 0.471 / 0.432 | 0.365 / 0.508 / 0.425 | 0.357 / 0.629 / 0.455 |
| 10h (60-day plot) | 462 / 87 / 78 | 0.612 / 0.446 / 0.516 | 0.598 / 0.506 / 0.548 | 0.451 / 0.594 / 0.512 |
| 13h (mixed plot) | 166 / 36 / 18 | 0.438 / 0.367 / 0.399 | 0.424 / 0.460 / 0.441 | 0.302 / 0.517 / 0.381 |


**Supplementary Table S10. The two polygon-trained models on the polygon test partition (50 images) under the same protocol. These values are not comparable with the main-text tables, which are computed on a different test partition under a different annotation convention, and each model was trained once only.**

| Model | AP50 | Weed recall (full) | F1 (two-class mean) | Accuracy | FPS |
|---|---|---|---|---|---|
| Mask R-CNN | 39.4 | 0.436 | 0.930 | 0.939 | 13.2 |
| YOLOv8-seg | 37.7 | 0.404 | 0.944 | 0.952 | 33.9 |


![fig4_training](/home/thanhtran/Gia-mia/mia/rae/figures/fig4_training.png)

Supplementary Figure S1. Training curves for the three bounding-box models, first training run (seed 0). Left axis: training loss; right axis, for YOLOv10: validation mAP@50. The abscissa is epochs for all three panels, the Detectron2 logs converted at 181 iterations per epoch; that framework computes no validation loss under this configuration. The training loss of all three is still falling at the final epoch, so none had reached a plateau, and the validation mAP@50 of YOLOv10 stops improving after roughly the tenth epoch. The break in the YOLOv10 curve marks two epochs at which the logged box-regression loss was non-finite; training recovered in the following epoch.

