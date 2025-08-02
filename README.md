# Semantic Segmentation for Directed Energy Deposition (DED)

## Overview

This repository contains the code and implementation details of a semantic segmentation model designed specifically for real-time process monitoring in Directed Energy Deposition (DED) additive manufacturing systems. The model leverages a Vision Transformer (ViT) encoder backbone and a Dense Prediction Transformer (DPT)-style decoder head to achieve pixel-wise classification, aiding precise process control and height estimation in additive manufacturing processes.

## Table of Contents

* [Introduction](#introduction)
* [Architecture](#architecture)

  * [Encoder Backbone](#encoder-backbone)
  * [Decoder](#decoder)
  * [Temporal Module](#temporal-module)
* [Dataset and Annotations](#dataset-and-annotations)
* [Installation](#installation)
* [Usage](#usage)

  * [Training](#training)
  * [Inference](#inference)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [License](#license)
* [Contact](#contact)

## Introduction

Directed Energy Deposition (DED) systems, widely used in aerospace and heavy industries, lack efficient real-time closed-loop process monitoring. Our approach addresses this limitation by employing a semantic segmentation model to monitor layer geometry and detect anomalies instantly, significantly reducing waste and improving manufacturing precision.

## Architecture

### Encoder Backbone

The model uses Meta’s DINOv2, a robustly pre-trained Vision Transformer (ViT) that extracts global context features. This backbone is capable of generalized feature extraction, ideal for transferring learning to the DED domain.

### Decoder

A custom Dense Prediction Transformer (DPT)-style decoder reconstructs spatially meaningful representations from encoder tokens to output dense semantic segmentation predictions. This decoder structure efficiently combines multi-scale transformer features.

### Temporal Module

To enhance segmentation consistency, the Temporal Deformable Dense Prediction Task (TDDPT) extends the model by including temporal memory tokens processed by an additional cross-attention mechanism. This significantly improves segmentation performance by incorporating temporal and spatial context.

## Dataset and Annotations

Our dataset comprises video sequences captured externally from DED machines. Segmentation annotations were generated efficiently using Meta’s SAM2 (Segment Anything Model), significantly reducing manual labeling efforts while providing accurate labels for the semantic segmentation training.

## Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/larson-7/ML4DED.git
cd ml4ded
pip install -r requirements.txt
```

## Data
Model weights, data labels, and raw videos can be found at the partner [HuggingFace repo](https://huggingface.co/iknocodes/ml4ded)

## Usage

### Training

To train the model:

```bash
python train.py --config configs/train_config.yaml
```

### Inference

For inference on new data:

```bash
python run.py --image-dir layer_images/5/ --expected-height-mm 4.79237 --device mps  --color-layers --enable-temporal
```

## Evaluation Metrics

* **Mean Intersection-over-Union (mIoU)**: Average accuracy across all classes.
* **Weighted mIoU**: Prioritizes critical classes such as the current deposition area.
* **Pixel Accuracy**: Overall prediction accuracy at the pixel level.

## Results

| Method          | mIoU (%) | Weighted mIoU (%) | Pixel Accuracy (%) |
| --------------- | ------- | ----------------- | ----------- |
| Baseline        | 78.4    | 73.5              | 97.4            |
| Proposed (ours) | 87.8     | 84.6               | 98.5            |

Initial experiments marked significant improvements in temporal consistency and segmentation accuracy with our proposed temporal attention mechanism.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contact

* **Jordan Larson** - [jlarson45@gatech.edu](mailto:jlarson45@gatech.edu)
* **Shean Scott Jr** - [sscott304@gatech.edu](mailto:sscott304@gatech.edu)

For further details, contributions, or collaboration inquiries, please contact the authors via email.
