# DLAV-2025 - Project

*by Aurélien Genin and Sara Heidaripour*

## Milestone 1 - Basic end-to-end planner

Our end-to-end planner uses three types of inputs from the [nuPlan](https://www.nuscenes.org/nuplan) dataset:
* ``camera``: RGB visual input from the camera at the time of inference (shape (200,300,3))
* ``history``: Past positions and headings (shape (21,3))
* ``command``: Driving command at the time of inference (shape (1), translated from ``[left,forward,right]`` to ``[-1,0,1]``)

It returns the output as a (60,3) array of the predicted future positions and headings.

### Network architecture

Our network first processes each type of input independantly, and then merges them to make the final prediction.

1. The ``camera`` input is processed using a simple CNN:
    * Conv2d(3, 16, kernel_size=5, stride=1, padding=2): 200x300x3 -> 200x300x16
    * ReLU
    * MaxPool2d(kernel_size=2): 200x300x16 -> 100x150x16
    * Conv2d(16, 32, kernel_size=5, stride=2, padding=2): 100x150x16 -> 50x75x32
    * ReLU
    * Flatten: 50x75x32 -> 50\*75\*32
    * Linear(50\*75\*32, 128): 50\*75\*32 -> 128
    * ReLU

2. The ``history`` input is processed using only one linear layer:
    * Flatten: 21x3 -> 21\*3
    * Linear(21\*3, 64): 21x3 -> 64
    * ReLU

Previous tests tried to use a transformer encoder layer to account for the temporal dimension, but this proved less precise than this simpler method.

3. The ``command`` input is processed but simply expanding it to give it more weight and ways of use for the inputs merger:
    * Linear(1, 8): 1 -> 8

4. Finally, the input merger network is a simple fully connected network:
    * Concatenate: 128,64,8 -> 128+64+8
    * Linear(128+64+8, 128+64+8): 128+64+8 -> 128+64+8
    * ReLU
    * Linear(128+64+8, 60\*3): 128+64+8 -> 60\*3

### Training

The training uses the MSE-loss with the Adam optimizer. Through different tests, the best training schedule was found to start with a large training rate, and then sequentially lower it. The final training schedule was: 
1. 5e-3 for 10 epochs
2. 3e-3 for 20 epochs
3. 2e-3 for 10 epochs
4. 5e-4 for 10 epochs

This method enabled to minimize over-fitting. Without it, the model would reach very low training loss, but keep a high validation loss. By sequentially lowering the training rate, the two losses remained aligned and lowered together.

The training batch size was kept to 32.

### Run the model

To run the model, simply use the attached [Jupyter notebook](DLAV_Phase1_Aurelien.ipynb). The model is loaded with the following commands.

```python
model = DrivingPlanner()
model.load_state_dict(torch.load("phase1_model.pth"))
```

## Milestone 2 - Perception-Aware Planning, Initial Architecture

Our perception-aware planner uses three types of inputs from the [nuPlan](https://www.nuscenes.org/nuplan) dataset:
* ``camera``: RGB visual input from the camera at the time of inference (shape (200,300,3))
* ``history``: Past positions and headings (shape (21,3))
* ``depth``: Depth image from the camera at the time of inference (shape (200,300,1))

It returns the output as a (60,3) array of the predicted future positions and headings.

### Network architecture

Our network first encodes each type of input independantly in a latent space, and then fuses them to make the final prediction.

1. The ``camera`` input is encoded using the pretrained EfficientNet B0 model [^eff_net]. We removed the last classification layers and replaced them with an AdaptiveAvgPool2d and a Linear layer to encode it to a 256-dimensions space.

2. The ``history`` input is processed using only linear layers:
    * Flatten: 21x3 -> 21\*3
    * Linear(21\*3, 256): 21x3 -> 256
    * ReLU
    * Linear(256, 128): 256 -> 128
    * LeakyReLU
  
3. The ``depth`` input is encoded using a CNN:
   * Conv2d(1, 32, kernel_size=3, stride=1, padding=1): 200x300x1 -> 200x300x32
   * ReLu
   * MaxPool2d(kernel_size=2): 200x300x32 -> 100x150x32
   * Conv2d(32, 64, kernel_size=5, stride=2, padding=2): 100x150x32 -> 50x75x64
   * ReLU
   * Conv2d(64, 64, kernel_size=5, stride=2, padding=2): 50x75x64 -> 25x38x64
   * ReLU
   * Flatten: 25x38x64 -> 25\*38\*64
   * Linear(25\*38\*64, 256): 25\*38\*64 -> 256
   * LeakyReLU
  
4. The fuser decoder network is a simple fully connected network:
    * Concatenate: 256,128,256 -> 256+128+256
    * Linear(256+128+256, 512): 256+128+256 -> 512
    * ReLU
    * Linear(512, 256): 512 -> 256
    * ReLU
    * Linear(256, 60\*3): 256 -> 60\*3
  
5. A depth encoder is used to re-generate the depth input and use an auxiliary loss:
    * ConvTranspose2d(1280, 128, kernel_size=4, stride=3, padding=1, output_padding=2): 1x1x1280 -> 22x31x128
    * ReLU
    * ConvTranspose2d(128, 32, kernel_size=4, stride=3, padding=1, output_padding=2): 22x31x128 -> 67x94x32
    * ReLU
    * ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1): 67x94x32 -> 134x188x1
    * Upsample(size=(200, 300), mode='bilinear', align_corners=False): 134x188x1 -> 200x300x1

### Loss

The training loss is built as a sum of a MSE loss on the future trajectory point and an L1 loss on the depth image. The auxiliary depth loss is used to train the network to "understand" what the depth input means, so that it structures the RGB camera latent space in a way that is pereception-aware.

| Output | Target | Loss | Weight |
| :----- | :----- | :--- | :----: |
| Predicted future positions | True future positions | MSE | 1 |
| Depth image from RGB image | True depth image | L1 | 0.05 |

### Training

Through different tests, the best training schedule was found to start with a large training rate, and then sequentially lower it. The final training schedule was: 
1. 2e-3 for 10 epochs
2. 1e-3 for 30 epochs
4. 5e-4 for 20 epochs
5. 3e-4 for 10 epochs

This method enabled to minimize over-fitting. Without it, the model would reach very low training loss, but keep a high validation loss. By sequentially lowering the training rate, the two losses remained aligned and lowered together.

The training batch size was set to 64.

## Milestone 2 - Perception-Aware Planning, Final Architecture (Inspired by CramNet)

Our planner integrates multi-modal data from the [nuPlan](https://www.nuscenes.org/nuplan) dataset to make trajectory predictions in a perception-aware and communication-efficient manner. The inputs are:
* ``camera``: RGB visual input from a forward-facing camera (shape: (200,300,3))
* ``depth``: Depth image from the same camera (shape: (200,300,1))
* ``semantic``: Semantic segmentation map (shape: (200,300,1))
* ``history``: Past 21 poses as (x, y, yaw) (shape: (21,3))

The output is a predicted sequence of 60 future positions (x,y) and headings (yaw): (60,3)

### Network architecture

Our model follows a modular, multi-stream architecture where each input modality is encoded separately, then refined using CramNet-inspired iterative fusion, and finally decoded into future trajectory predictions. The network includes an auxiliary depth decoder to encourage rich feature representations.

1. ``Camera Encoder (RGB)``
    * Backbone: EfficientNet-B0 pretrained on ImageNet
    * Final layers replaced by:
         * AdaptiveAvgPool2d
         * Flatten
         * Linear to 256-dim embedding
    * Output shape: (256,)
2. ``Depth Encoder``
    * CNN stack:
         * Conv2d(1, 32, 3, padding=1), ReLU
         * MaxPool2d(2)
         * Conv2d(32, 64, 5, stride=2, padding=2), ReLU
         * Conv2d(64, 64, 5, stride=2, padding=2), ReLU
         * Flatten, Linear to 256-dim
    * Output shape: (256,)
3. ``Semantic Encoder``
     Same structure as the depth encoder (independent weights). Encodes semantic maps into a 256-dim vector.
4. ``History Encoder``
   * Flatten: 21x3 → 63
   * Linear(63, 256), ReLU
   * Linear(256, 128), LeakyReLU
   * Output shape: (128,)
5. ``CramNet-Inspired Iterative Fusion Layer``
       Inspired by CramNet, our model uses iterative message passing across modalities instead of a single fusion step:
       * Inputs: RGB, Depth, Semantic (each 256-dim)
       * At each iteration:
          * Cross-modal attention or interaction between modalities
          * Residual updates
          * Fusion via MLP layers
       * Number of iterations: configurable (default: 3)
       * Output: fused latent vector (256-dim)
       This approach helps modalities "cram" knowledge into one another across multiple steps, enabling efficient representation learning with minimal capacity loss.
6. ``Planner Decoder``
    * Concatenate: [fused features (256), history (128)] → (384)
    * Fully connected layers:
         * Linear(384, 512), ReLU
         * Linear(512, 256), ReLU
         * Linear(256, 60*3)
7. ``Auxiliary Depth Decoder``
    A convolutional decoder reconstructs the depth map from the fused visual features:
    * Linear(256, 25×38×64) → reshape → (64, 25, 38)
    * ConvTranspose2d → Upsample to original size
    * Final output: (1, 200, 300)

### Loss

We use a multi-task loss to guide learning:

| Output | Target | Loss | Weight |
| :----- | :----- | :--- | :----: |
| Predicted future trajectory | True trajectory | MSE | 1.0 |
| Reconstructed depth map | Ground truth depth | L1 | 0.05 |

The auxiliary loss ensures that the RGB encoder retains depth-aware representations, enhancing perception awareness.

### Training

The training procedure uses progressive learning rate decay to avoid overfitting:
1. 2e-3 for 10 epochs
2. 1e-3 for 30 epochs
4. 5e-4 for 15 epochs
5. 1e-5 for 20 epochs
6. 5e-6 for 15 epochs

Batch size: 64
Optimizer: AdamW

### Summary of CramNet Inspiration
This architecture is directly inspired by CramNet’s key ideas:
* Iterative cross-modal fusion: Rather than concatenating all modality features once, we iteratively update each feature with others’ information.
* Lightweight late fusion: Encourages modality-specific encoders, and promotes efficient knowledge transfer through repeated message passing.
* Compact feature exchange: Each modality learns to compress its key knowledge into a shared latent space without overloading the decoder.
This design achieves better multi-modal alignment, resilience to missing modalities, and strong generalization for planning tasks.

### Run the model

To train and run the model, simply use the attached [Jupyter notebook](DLAV_Phase2_FinalVersion_CramNet_Inspired.ipynb)

## Milestone 3 - Sim-to-real Generalization

Our end-to-end planner for the real data uses only the ``camera`` and ``history`` inputs, and returns the future trajectory.

### Network architecture

Our network uses exactly the same architecture as Phase 2, without the depth encoder and decoder as we don't have these inputs available.

### Training

The training here is simpler: 20 epochs at 1e-3 learning rate. However, to perform well on real data, we used a mixed training set with real data but also simulated data as in the previous phases. This enabled us to reach good ADE result on real data.

### Run the model

To train and run the model, simply use the attached [Jupyter notebook](DLAV_Phase3_Aurelien-Sara.ipynb)

## References

[^eff_net]: M. Tan, and Q. V. Le, ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946), arXiv, 2019
