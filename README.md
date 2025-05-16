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

## Milestone 2 - Perception-Aware Planning

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

### Run the model

To train and run the model, simply use the attached [Jupyter notebook](DLAV_Phase2_Aurelien-Sara.ipynb)

## References

[^eff_net]: M. Tan, and Q. V. Le, ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946), arXiv, 2019
