# DLAV-2025 - Project - Phase 1

*by Aurélien Genin and Sara Heidaripour*

## Basic end-to-end planner

Our end-to-end planner uses three types of inputs from the [nuPlan](https://www.nuscenes.org/nuplan) dataset:
* ``camera``: RGB visual input from the camera at the time of inference (shape (200,200,3))
* ``sdc_history_feature``: Past positions and headings (shape (21,3))
* ``driving_command``: Driving command at the time of inference (shape (1), translated from ``[left,forward,right]`` to ``[-1,0,1]``)

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

2. The ``sdc_history_feature`` input is processed using only one linear layer:
    * Flatten: 21x3 -> 21\*3
    * Linear(21\*3, 64): 21x3 -> 64
    * ReLU

Previous tests tried to use a transformer encoder layer to account for the temporal dimension, but this proved less precise than this simpler method.

3. The ``driving_command`` input is processed but simply expanding it to give it more weight and ways of use for the inputs merger:
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

To run the model, simply use the attached Jupyter notebook. The model is loaded with the following commands.

```python
model = DrivingPlanner()
model.load_state_dict(torch.load("phase1_model.pth"))
```