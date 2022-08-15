
# Monocular Depth Estimation Literature Reviews

This is a set of brief reviews in the **Monocular Depth Estimation** area on most recent research papers in the filed. 

Monocular Depth Estimation is the task of estimating the depth value (distance relative to the camera) of each pixel given a single (monocular) RGB image. This challenging task is a key prerequisite for determining scene understanding for applications such as 3D scene reconstruction, autonomous driving, and AR.



## Pure MDE

Here, summarize the investigations in the new useful MDE softwares which are included in a research project. I mean they have the both open-source code and the paper.

After presenting the aforementioned data about each paper, answer to the following questions in your review:

- Explain an outline of the method

- Describe the performance in terms of depth estimation quality, FPS, etc. such that makes it comparable with the rest of the similar works

*TIP:* Here, we need to make a list of the open-source MDE state-of-the-art. Make sure that the papers you list all have a source code link. We'll need to deploy one, or a couple of the best ones.
To complete this part, see the following link(s):

[link](https://paperswithcode.com/task/monocular-depth-estimation)

And also you can do your own investigation to get the most valid softwares and enhance the list.


## MDE for SLAM

Here, talk about the research projects which their output was a SLAM base-on, or aided-by the MDE.
In your review, answer to the following questions:

- Explain an outline of the method

- Is there any sensor fusion or data fusion? What are the sensors? Is the system V-INS? Just Visual? Or something else?

- If used, how was the neural network architecture?

- Give a criteria of the calculational ability (FPS in terms of processor)

- In what range of cases, and how, is the system implemented? What were the mission and mission specifications?

- Is the system based on a flying robot platform? If yes, what was the platform? 


### Pseudo RGB-D for Self-improving Monocular SLAM and Depth Prediction

[Link to the Paper](https://drive.google.com/file/d/1SH6aLfpXGzQPGlCQWy0Moy6ERl6ExXaH/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately


### SDF-SLAM: A Deep Learning Based Highly Accurate SLAM Using Monocular Camera Aiming at Indoor Map Reconstruction With Semantic and Depth Fusion

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
CHEN YANG, QI CHEN, YAOYAO YANG, JINGYU ZHANG, MINSHUN WU, AND KUIZHI MEI

**Date:** 
Received December 31, 2021, accepted January 12, 2022, date of publication January 19, 2022, date of current version January 28, 2022

**Journal or Conference:** ...
IEEE journal

#### Review:

Type a paragraph

#### Answers:

1. Outline: A) They design and implement a semantic and deep fusion convolutional neural network (SDFCNN), which can perform semantic segmentation and depth estimation simultaneously by inferring input color RGB images. It greatly reduces the number of parameters, computation and time required for inferring the two network models.
B) They propose the feature point and feature description convolutional neural network (FPFDCNN) for feature point extraction from two different frames of images and generate vectors as descriptors for each feature point. FPFDCNN adopts the pixel shuffle algorithm to carry out superpixel restoration on low-resolution images. Using FPFDCNN can significantly reduce the environmental impact and finally generate high-resolution images.
C) They also add a data correction module to optimize a point cloud map globally to establish a consistent point cloud map and to greatly reduce the inference time and network parameters.

2. They construct a semantic and depth fusion SLAM (SDF-SLAM) framework, which fuses camera pose information and depth and semantic information of each frame.

3. **SDF-SLAM architecture:**

![Untitleddd](https://user-images.githubusercontent.com/106483656/184733920-e52d382b-fcd1-4d81-859b-035086f41fc3.jpg)

4. **Results:** The average accuracy of the predicted point cloud coordinates reaches 90%, and the average accuracy of the semantic labels reaches 67%. Moreover, compared with the state-of-the-artSLAMframeworks, such as ORB-SLAM, LSD-SLAM, and CNN-SLAM, the absolute error of the camera trajectory on indoor data with more feature points is reduced from 0.436 m, 0.495 m, and 0.243 m to 0.037 m, respectively. On indoor data with fewer feature points, they decrease from 1.826 m, 1.206 m, and 0.264 m to 0.124 m, respectively.

5. 


6. No its not.

## MDE for VIO and Navigation

Summarize the works concentrating on obtaining navigation data and visual odometry.

In your review, answer to the following questions:

- Explain an outline of the method

- Is there any sensor fusion or data fusion? What are the sensors? Is the system V-INS? Just Visual? Or something else?

- If used, how was the neural network architecture?

- Give a criteria of the calculational ability (FPS in terms of processor)

- In what range of cases, and how, is the system implemented? What were the mission and mission specifications?

- Is the system based on a flying robot platform? If yes, what was the platform? 


### Towards Scale-Aware, Robust, and Generalizable Unsupervised Monocular Depth Estimation by Integrating IMU Motion Dynamics

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately


### M4Depth: Monocular depth estimation for autonomous vehicles in unseen environments

[Link to the Paper]()

[Link to the Source Code](https://github.com/michael-fonder/M4Depth)

**Authors:** 
Michaël Fonder, Damien Ernst, and Marc Van Droogenbroeck

**Date:** 
last revised 1 Jul 2022

**Journal or Conference:** 

#### Answers:
1. Outline: 1) They extend the notion of visual disparity to camera baselines featuring six degrees of freedom (6 DoF) transformations, and present customized cost volumes for this disparity. 2) They present a novel real-time and lightweight multi-level architecture based on these cost volumes to perform end to end depth estimation on video streams acquired in unstructured environments. 3) It is shown that M4Depth, is state of the art on the Mid-Air dataset, that it has good performances on the KITTI dataset, that it outperforms existing methods in a generalization setup on the TartanAir dataset.

2. Sensor fusion is not used. Just visual. Pseudo sensor: MDE

3. Architecture overview of M4Depth (with three levels here):

![Untitled](https://user-images.githubusercontent.com/106483656/184727447-435cab8f-1e5d-4f4c-9f9f-17404997a0c1.jpg)

It is fed by two consecutive frames and the camera motion. Each disparity refiner produces a disparity estimate and learnable disparity features. All convolutions are followed by a leaky ReLU activation unit, except for the ones producing a disparity estimate. To ease the convergence, disparities are encoded in the log-space.
They use these new cost volumes(the Disparity Sweeping Cost Volume (DSCV) and the Spatial Neighborhood Cost Volume (SNCV) ) to leverage the visual spatio-temporal constraints imposed by motion and to make the network robust for varied scenes.

4. Performance metrics are taken from Eigen et al. for depth maps limited to 80 m.
This network has 4.5 million parameters and requires up to 500 MB of GPU memory to run with six levels. At inference time on Mid-Air, an NVidia Tesla V100 GPU needs 17 ms to process a single frame for a raw TensorFlow implementation. This corresponds to 59 fps which is roughly twenty-times faster than DeepV2D, the best performing method on KITTI. Such inference speed is compatible with the real-time constraints required for robotic applications.

5.  In order to compare their method with its parent, PWC-Net, they perform the same experiments for both M4Depth and PWC-Net.
They present three experiments: 
A) They compare the performance of their method with the ones of the state of the art on a dataset featuring unstructured environments, using Mid-Air dataset.
B) They assess the performance on a standard depth estimation using KITTI dataset.
C) They evaluate the generalization capabilities of all the methods. They use static scenes that are semantically close to either the Mid-Air or the KITTI dataset, and test the performance of the method trained on Mid-Air (respectively KITTI) on the selected unstructured (respectively urban) scenes without any fine-tuning. For this experiment, they use TartanAir dataset.

6. No, its not.

7. **Future works:** Their further works on M4Depth will, among others, focus on the determination of its own uncertainty on depth estimates at inference time. Such an addition would provide a great advantage over other methods that do not offer this capability.

8. **Performance:** Their network outperforms the state of the art on these datasets. M4Depth is superior to the baseline both in unstructured environments and in generalization while also performing well on the standard KITTI benchmark, which shows its superiority for autonomous vehicles. In addition to being motion- and featureinvariant, their method is lightweight, runs in real time, and can be trained in an end-to-end fashion.

### Bayesian cue integration of structure from motion and CNN‑based monocular depth estimation for autonomous robot navigation

[Link to the Paper](https://drive.google.com/file/d/1s-s11aPrmPC9uOqdnJ91yBntFpcqAdvS/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately


### SelfVIO: Self-supervised deep monocular Visual–Inertial Odometry and depth estimation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately


### Multi-Sensor Fusion Self-Supervised Deep Odometry and Depth Estimation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately


### On deep learning techniques to boost monocular depth estimation for autonomous navigation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately


### Joint Estimation of Depth and Pose with IMU-assisted Photometric Loss

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

Type the answers separately

