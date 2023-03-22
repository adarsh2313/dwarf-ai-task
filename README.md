# dwarf-ai-task

In this task to predict the height of the child using key point coordinates from depth map, I have took the approach of creating a simple neural network with 3 layers:
1. Input layer of 9 input features - y co-ordinates of the following keypoints
a. Nose
b. Left shoulder
c. Right shoulder
d. Left hip
e. Right hip
f. Left knee
g. Right knee
h. Left ankle
i. Right ankle

2. Hidden layer of 5 nodes
3. Output layer of 1 node, predicting the height of the child

The reason for incorporating a Deep learning approach with a simple neural network is that there will be several variation to the images of children obtained ranging from orinetation, posture, etc which traditional ML models will fail to take these into account.
For example, the child may be bending down or standing sideways, even then the model should accurately predict the height of the child.

The data consists of various input features and out of those, I found the above 9 to be of utmost importance. I took the y coordinates of the above coordinates as it will account more to the height of the child while x coordinate has little to no role to play in evaluating the height.
