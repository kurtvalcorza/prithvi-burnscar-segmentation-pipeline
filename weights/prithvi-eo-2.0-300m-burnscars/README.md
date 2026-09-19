---
license: apache-2.0
library_name: terratorch
---

### Model and Inputs
The pretrained [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) model is fine-tuned to segment the extent of burned areas on HLS images from the [HLS Burn Scars dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars). 

The dataset consists of ~800 labeled 512x512 chips from the continental US.  

We use the following six bands for the predictions: Blue, Green, Red, Narrow NIR, SWIR, SWIR 2.

Labels represent no burned areas (class 0), burned areas (class 1), and no data/clouds (class -1).

The Prithvi-EO-2.0-300M model was initially pretrained using a sequence length of 4 timestamps. Based on the characteristics of this benchmark dataset, we focus on single-timestamp segmentation. 
This demonstrates that our model can be utilized with an arbitrary number of timestamps during fine-tuning.

### Fine-tuning

The model was fine-tuned using [TerraTorch](https://github.com/IBM/terratorch) which you can install with `pip install terratorch`. For fine-tuning run:

```shell
terratorch fit -c burn_scars_config.yaml
```

The configuration used for fine-tuning is available in [burn_scars_config.yaml](burn_scars_config.yaml).

We created new non-overlapping splits for train, validation and test which you find in [splits](splits).
The same splits where used in the evaluation in the Prithvi-EO-2.0 paper. 
Compared to the Prithvi-EO-2.0 paper, we used a UNetDecoder instead of a UperNetDecoder for this model. 
We repeated the run five times and selected the model with the lowest validation loss over all runs and epochs.
Finally, we evaluated the selected model on the test split with the following results: 

| Model               | Decoder     | test IoU Burned | test mIoU | val IoU Bruned | val mIoU |
|:--------------------|:------------|:---------------:|:---------:|:--------------:|:--------:|
| Prithvi EO 2.0 300M | UNetDecoder |      87.52      |   93.00   |     84.28      |  90.95   |


### Inference and demo

A **demo** running this model is available **[here](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-BurnScars-demo)**.

This repo includes an inference script that allows running the model for inference on HLS images.

```shell
python inference.py --data_file examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif
```

### Feedback

Your feedback is invaluable to us. If you have any feedback about the model, please feel free to share it with us. You can do this by submitting issues on GitHub or start a discussion on HuggingFace.

### Citation

If this model helped your research, please cite [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) in your publications.

```
@article{Prithvi-EO-V2-preprint,    
    author          = {Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and Gíslason, Þorsteinn Elí and Blumenstiel, Benedikt and Ghosal, Rinki and de Oliveira, Pedro Henrique and de Sousa Almeida, João Lucas and Sedona, Rocco and Kang, Yanghui and Chakraborty, Srija and Wang, Sizhe and Kumar, Ankur and Truong, Myscon and Godwin, Denys and Lee, Hyunho and Hsu, Chia-Yu and Akbari Asanjan, Ata and Mujeci, Besart and Keenan, Trevor and Arévolo, Paulo and Li, Wenwen and Alemohammad, Hamed and Olofsson, Pontus and Hain, Christopher and Kennedy, Robert and Zadrozny, Bianca and Cavallaro, Gabriele and Watson, Campbell and Maskey, Manil and Ramachandran, Rahul and Bernabe Moreno, Juan},
    title           = {{Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications}},
    journal         = {arXiv preprint arXiv:2412.02732},
    year            = {2024}
}
```