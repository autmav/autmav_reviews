
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
Lokender Tiwari, Pan Ji, Quoc-Huy Tran, Bingbing Zhuang, Saket Anand, and Manmohan Chandraker

**Date:** 
Submitted on 22 Apr 2020 (v1), last revised 7 Aug 2020 (v3)

**Journal or Conference:**
arXiv

#### Review:

They propose a joint narrow and wide baseline based self-improving framework to couple geometrical and learning based methods for 3D perception, where on the one hand the CNN-predicted depth is leveraged to perform pseudo RGB-D feature-based SLAM, leading to better accuracy and robustness than the monocular RGB SLAM baseline.
Their framework only requires unlabeled monocular videos in both training and inference stages, and yet is able to outperform state-of-the-art self-supervised monocular and stereo depth prediction networks (e.g., Monodepth2) and feature-based monocular SLAM system (i.e., ORB-SLAM).

#### Answers:

1. They introduce two wide baseline losses, i.e., the symmetric depth transfer loss and the depth consistency loss on common tracked points, and propose a joint narrow and wide baseline based depth prediction learning setup, where appearance based losses are computed on narrow baselines and purely geometric losses on wide baselines (non-consecutive temporally distant keyframes).

2. Pseudo RGB-D + SLAM
Their fusion of geometric SLAM and CNN-based monocular depth estimation turns out to be symbiotic and this complementary nature sets the basis of their self-improving framework.

3. Network type: CNN
**Overview of Self-Improving Framework:**

![Untitled](https://user-images.githubusercontent.com/106483656/187045870-25263a3c-ef04-40e4-bbf2-7b4419ee7700.jpg)

4. FPS information is not mentioned.
A single self-improving loop takes 0:6 hour on a *NVIDIA TITAN Xp 8GB GPU*.

5. **Implementation:** They implement their framework based on Monodepth2 and ORB-SLAM, i.e., they use the depth network of Monodepth2 and the RGB-D version of ORB- SLAM for depth refinement and pose refinement respectively.

**KITTI Eigen Split/Odometry Experiments:** They pre-train MonoDepth2 using monocular videos of the KITTI Eigen split training set with the hyperparameters as suggested in MonoDepth2.

**TUM RGB-D Experiments:** For TUM RGB-D, They pre-train/fine-tune the depth network on 2 freiburg3 sequences, and test on 2 freiburg3 sequences.

**Metrics for Pose Evaluation:** Root Mean Square Error (RMSE), Relative Translation (Rel Tr ) error, and Relative Rotation (Rel Rot) error of the predicted camera trajectory. 

**Metrics for Depth Evaluation.:** They use the standard metrics, including the Absolute Relative (Abs Rel ) error, Squared Relative (Sq Rel ) error, RMSE, RMSE log, Delta<1.25 (namely a1 ), Delta<1.25^2 (namely Delta^2 ), and Delta<1.25^3 (namely Delta^3 )

**Results:** Through extensive experiments on *KITTI and TUM RGB-D* datasets, their framework is shown to outperform both monocular SLAM system (i.e., ORB-SLAM) and the state-of-the-art unsupervised single-view depth prediction network (i.e., Monodepth2).

6. No it's not.

7. **Future works:** Currently, their self-improving framework only works in an off-line mode, so developing an on-line real-time self-improving system remains one of their future works. Another avenue for their future works is to move towards more challenging settings, e.g., rolling shutter cameras or uncalibrated cameras.

### SDF-SLAM: A Deep Learning Based Highly Accurate SLAM Using Monocular Camera Aiming at Indoor Map Reconstruction With Semantic and Depth Fusion

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
CHEN YANG, QI CHEN, YAOYAO YANG, JINGYU ZHANG, MINSHUN WU, AND KUIZHI MEI

**Date:** 
Received December 31, 2021, accepted January 12, 2022, date of publication January 19, 2022, date of current version January 28, 2022

**Journal or Conference:**
IEEE journal

#### Answers:

1. **Outline:** A) They design and implement a semantic and deep fusion convolutional neural network (SDFCNN), which can perform semantic segmentation and depth estimation simultaneously by inferring input color RGB images. It greatly reduces the number of parameters, computation and time required for inferring the two network models.
B) They propose the feature point and feature description convolutional neural network (FPFDCNN) for feature point extraction from two different frames of images and generate vectors as descriptors for each feature point. FPFDCNN adopts the pixel shuffle algorithm to carry out superpixel restoration on low-resolution images. Using FPFDCNN can significantly reduce the environmental impact and finally generate high-resolution images.
C) They also add a data correction module to optimize a point cloud map globally to establish a consistent point cloud map and to greatly reduce the inference time and network parameters.

2. They construct a semantic and depth fusion SLAM (SDF-SLAM) framework, which fuses camera pose information and depth and semantic information of each frame.

3. They propose a CNN named FPFDCNN for feature point extraction. The FPFDCNN model consists of five components, including input layer, encoder layers, decoder layers, output layers and concatenate layer. FPFDCNN network structure:

![FPFDCNN network structure](https://user-images.githubusercontent.com/106483656/185100051-5b8fbe59-695f-4bab-b486-df10a08d4686.jpg)

4. **Results:** The average accuracy of the predicted point cloud coordinates reaches 90%, and the average accuracy of the semantic labels reaches 67%. Moreover, compared with the state-of-the-artSLAMframeworks, such as ORB-SLAM, LSD-SLAM, and CNN-SLAM, the absolute error of the camera trajectory on indoor data with more feature points is reduced from 0.436 m, 0.495 m, and 0.243 m to 0.037 m, respectively. On indoor data with fewer feature points, they decrease from 1.826 m, 1.206 m, and 0.264 m to 0.124 m, respectively.
The camera pose obtained by SDF-SLAM, which extracts feature points through a CNN, is more accurate than that obtained by ORB-SLAM.

5. **SDF-SLAM experimental platform:**

![SDF-SLAM experimental platform](https://user-images.githubusercontent.com/106483656/185103114-9c66d78c-97f2-419d-9169-f8753161e7ea.jpg)

6. No its not.

7. **Ideas:** Currently, the state-of-art methods use feature point matching to estimate camera pose based on the PnP method.

8. **Datasets:** They select the rgbd_dataset_freiburg1_desk2 (f1_desk2) scene from the TUM dataset for feature point extraction named FP dataset.

9. **SDF-SLAM architecture:**

![Untitleddd](https://user-images.githubusercontent.com/106483656/184733920-e52d382b-fcd1-4d81-859b-035086f41fc3.jpg)

10. **3-Dimensional semantic map's steps:** (1) Design a deep semantic fusion CNN to complete the two tasks of image depth estimation and semantic segmentation. (2) Fuse the depth information and semantic information into 3D semantic point cloud data. (3) Map the point cloud data of each frame to the world coordinate system according to the camera pose to obtain a 3D semantic map.

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
Sen Zhang , Jing Zhang , and Dacheng Tao

**Date:** 
21 Jul 2022

**Journal or Conference:**
arXiv

#### Review:

They propose DynaDepth, a scale-aware, robust, and generalizable MDE framework using IMU motion dynamics.

#### Answers:

1. They first propose an IMU photometric loss and a cross-sensor photometric consistency loss to provide dense supervision and absolute scales. To fully exploit the complementary information from both sensors, they further drive a differentiable camera-centric *extended Kalman filter (EKF)* to update the IMU preintegrated motions when observing visual measurements.

**Overall Framework:** 

![Untitledddd](https://user-images.githubusercontent.com/106483656/186728218-d7299a7b-b7f7-4478-a101-be2211c5347d.jpg)

2. **Sensors:** IMU photometric loss + cross-sensor photometric consistency loss +

**Sensor fusion:** They derive a camera-centric EKF framework for the sensor fusion, which also provides an egomotion uncertainty measure under the setting of unsupervised learning. 

3. It is not mentioned.

4. FPS informations is not mentioned. The training process takes 1 ~ 2 days on a single NVIDIA V100 GPU.

5. 
* They compare DynaDepth with state-of-the-art MDE methods which rescale the results using the ratio of the median depth between the ground-truth and the prediction. DynaDepth achieves the best up-to-scale performance w.r.t. four metrics and achieves the second best for the other three metrics. Of note is that DynaDepth also achieves a nearly perfect absolute scale.
* Then they compare the unscaled results with PackNet- SfM and G2S, which use the GPS information. DynaDepth achieves the best performance w.r.t. all metrics.
* They test the generalizability of DynaDepth on Make3D using models trained on KITTI. They found that DynaDepth that only uses the gyroscope and accelerator IMU information achieves the best generalization results.

**Implementation:** DynaDepth is implemented in pytorch. They adopt the monodepth2 network structures. (ps:The source codes and the trained models will be released)

**Results:** By leveraging IMU during training, DynaDepth not only learns an absolute scale, but also provides a better generalization ability and robustness against vision degradation such as illumination change and moving objects.

**Ablation Studies:** It's on KITTI to investigate the effects of the proposed IMU-related losses, the EKF fusion framework, and the learnt ego-motion uncertainty. Also they design simulated experiment to demonstrate the robustness of DynaDepth against vision degradation such as illumination change and moving objects. WLOG, we use ResNet18 as the encoder for all ablation studies.

6. No it's not.

7. **DynaDepth benefits:** (1) the learning of the absolute scale, (2) the generalization ability, (3) the robustness against vision degradation such as illumination change and moving objects, and (4) the learning of an ego-motion uncertainty measure, which are also supported by our extensive experiments and simulations on the KITTI and Make3D datasets.

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
Fuseini Mumuni, Alhassan Mumuni

**Date:** 
Published online: 2 March 2022

**Journal or Conference:** ...
Springer Nature Singapore Pte Ltd. 2022

#### Review:

They exploit structure from motion (SfM) through optical flow as an additional depth cue and prior knowledge about depth distribution in the environment to improve monocular depth prediction to enable robot navigation in an unstructured, hitherto unfamiliar environment.. They show how it is possible to promote MDE cue from ordinal scale to the same metric scale as SfM, thus, enabling their optimal integration in a Bayesian optimal manner.

#### Answers:

1. **Overall workflow:**

![Workflow](https://user-images.githubusercontent.com/106483656/185744336-02035a7a-6492-4937-8f8f-e5c3010e41fc.jpg)

2. The captured frame is processed using the MDE and SfM models described, and together with the speed and pose information, depth fusion is performed. Also sensory cues from MDE and SfM are fused to reduce the resulting variance (standard deviation).
Sensor fusion algorythm: optimized scaling adaptive Kalman Filter (osAKF)
Sensor/Pseudo sensors: IMU + Camera + Lidar(for prior experiments) + Odometry + CNN‑based MDE + SfM 

3. **Proposed Network Architecture:**

![Untitled2](https://user-images.githubusercontent.com/106483656/185747128-da6aa709-c7ed-4e4f-9452-06cc74a48841.jpg)

The neural network architecture proposed in this work is inspired by Alhashim and Wonka, whose CNN architecture consists of an encoder based on transfer learning, and a simple decoder. The feature extractor in the said work is a pretrained DenseNet-169 model using the ImageNet dataset. The encoder of their model is a custom network pre-trained on Kaggle’s Cats vs Dogs dataset. The network uses Max pooling and ReLU activations throughout.
The network uses a weighted combination of four losses—mean absolute depth error (ADE), the scale-invariant (S-I) function proposed by Lee et al. , image gradient (IG) loss and structural similarity (SSIM) loss function for training. They used ADE, S-I and IG functions as defined in Alhashim and Wonka.
The architecture greatly reduces model complexity and ensures efficiency.

4. The data rates for pose and speed information are 200 Hz and 100 Hz, respectively.

5. 
* Sensor models (likelihoods of MDE and SfM sensor model) and prior are determined.
* They built a model representing this relationship by first running a series of experiments for various combinations of MDE predicted depth, versus actual scene depths corresponding to various camera tilt angles only for corresponding image regions. (from 0 to 45° and 0 to - 45° with increments of 5°. It is limited to 45° since larger angles produce too narrow spatial overlap). As the relative tilt between two frames increases, the overlapping region of their field of views decreases.
* two conditions were of interest—the ability to perceive depth accurately with (a) normal camera pose and (b) camera subjected to angular tilts.

**Training:** Training of the network is performed in two stages, as illustrated in Algorithm 1. The weights for the part of the encoder based on pretrained Cats vs Dogs classifier are frozen during the first round of training (i.e., when weight update mode is “partial”). Meanwhile, the weights for the second part of encoder, as well as the decoder, are initialized using *He Normal* distributions. In the second round of training—when weight update mode is “all layers”—all network weights are updated.

6. No it's not. It is *Robust Autonomous Indoor Navigation robot (RAINav)*; a custom-built mini vehicle system with a camera in the front face of the vehicle and four motors interfaced with OMRON E6B2-CWZ6C high-accuracy rotary encoders. 

**Major components of the robot:** 1—Jetson TX2; 2—Arm Cortex M4 microcontroller system; 3—Camera; 4—LiDAR (only used for sampling prior depth values) 

7. **Future works:** Future work could incorporate object recognition as well as size and rigidity estimation pipelines to distinguish objects that should be avoided, from small or deformable ones on the robot’s path that could be trampled on.

8. **Performance:** They have been able to improve depth estimation accuracy by more than 6% in a mobile robot navigation setting where the robot’s heading can under large angular inclinations.
Experimental results show that besides providing metric scale for dense depth map, the integration of these cues significantly improves depth accuracy, reliability and robustness beyond what could be obtained from monocular vision or structure from motion alone.

### SelfVIO: Self-supervised deep monocular Visual–Inertial Odometry and depth estimation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
Yasin Almalioglu, Mehmet Turan, Alp Eren Sari, Muhamad Risqi U. Saputra, Pedro P. B. de Gusmão, Andrew Markham, Niki Trigoni

**Date:** 
Last revised 23 Jul 2020 

**Journal or Conference:** ...
 ُSubmitted to The IEEE Transactions on Robotics (T-RO) journal, under review
 
#### Review:

They introduce a novel self-supervised deep learning-based VIO and depth map recovery approach (SelfVIO) using adversarial training and self-adaptive visual-inertial sensor fusion.
* No strict temporal or spatial calibration between camera and IMU is necessary for pose and depth estimation, contrary to traditional VO approaches.
* To the best of their knowledge, this is the first selfsupervised deep joint monocular VIO and depth reconstruction method in the literature.

#### Answers:

1. **Architecture overview:**

![Untitled](https://user-images.githubusercontent.com/106483656/185489464-606c325d-2f64-4e0f-9d7c-59f99e1823d3.jpg)

Unlabeled image sequences and raw IMU measurements are provided as inputs to the network. The method estimates relative translation and rotation between consecutive frames parametrized as 6-DoF motion and a depth image as a disparity map for a given view.

2. They propose a novel unsupervised sensor fusion technique for the camera and the IMU, which extracts and fuses motion features from raw IMU measurements and RGB camera images using convolutional and recurrent modules based on an attention mechanism.

3. **Pose estimation and depth map generation architecture:**
 
![Untitledd](https://user-images.githubusercontent.com/106483656/185490839-e59df649-6106-4b6e-ba3f-74c555658a6d.jpg)

4.

5. They demonstrated superior performance of SelfVIO against state-of-the-art VO, VIO, and even VSLAM approaches on the KITTI, EuRoC and Cityscapes datasets.

6. No it's not.

7. **Future works:** In future work, they plan to develop a stereo version of SelfVIO that could utilize the disparity map.

### Multi-Sensor Fusion Self-Supervised Deep Odometry and Depth Estimation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
YingcaiWan, Qiankun Zhao, Cheng Guo, Chenlong Xu and Lijing Fang

**Date:** 
Received: 16 December 2021 / Revised: 10 February 2022 / Accepted: 14 February 2022 / Published: 2 March 2022

**Journal or Conference:** ...
MDPI journal

#### Review:

They present a new deep visual-inertial odometry and depth estimation framework for improving the accuracy of depth estimation and ego-motion from image sequences and inertial measurement unit (IMU) raw data. The proposed framework predicts ego-motion and depth with absolute scale in a self-supervised manner that combines DepthNet with DeepVIO to supervise each other.

#### Answers:

1. Based on the SuperPoint dense feature point extraction method, they added the sparse depth pose with absolute scale to the depth estimation geometric constraints; The DeepVIO pipeline joint keypoint is based on DVO with DIO and uses the EKF module to update the relative pose.

2. Sensors/pseudo sensors: New deep visual-inertial odometry and depth estimation: + deep visual odometry (DVO) + deep inertial odometry (DIO) + deep visual-inertial odometry (DeepVIO)
**Data fusion:** By extended Kalman filter (EKF)

3. An end-to-end self-supervised learning architecture

4.

5. They tested their framework on the KITTI dataset, showing that their approach produces more accurate absolute depth maps than contemporaneous methods. Their model also demonstrates stronger generalization capabilities and robustness across datasets.

6.

### On deep learning techniques to boost monocular depth estimation for autonomous navigation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:** ...

#### Review:

Type a paragraph

#### Answers:

1. 

2.

3.

4.

5.

6.

### Joint Estimation of Depth and Pose with IMU-assisted Photometric Loss

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
Xiaohui Yang; Yu Feng; Jian Li; Zhiqing Chen; zhenping Sun; Xiaopu Nie

**Date:** 
Date of Conference: 27-28 November 2020
Date Added to IEEE Xplore: 07 December 2020

**Journal or Conference:** 
Published in 2020 3rd International Conference on Unmanned Systems (ICUS) by IEEE

#### Review:

A) When jointly estimating depth and pose by minimizing photometric loss, we introduce IMU, named as Pose Hints to provide suggestions for pose. As a result, they get better results when compared with their baseline network;
B) They get the pseudo-IMU from consecutive poses and compare it with Pose Hints to measure the accuracy of pose optimization and then promote the optimization process based on this measurement.

#### Answers:

1. They combine CodeSalm and BANet to set up their baseline method. It consists of a Depth Map Generator and a Nonlinear Optimization Module. In training phase, the Depth Map Generator is trained in supervised manner and in inference phase, the Nonlinear Optimization Module jointly reoptimize pose and depth.

2.  Pose Hints provide pose suggestions during optimization phase, so to fuse the two heterogeneous data, they embed a nonlinear optimization module in CNNs.
Sensors/pseudo sensors: IMU

3. **Overview of network structure:**
In inference phase, the Depth Map Generator first generates a set of basis depth maps B and the initial depth D_init, then the Nonlinear Optimization Module takes B, D_init and consecutive frames as inputs to reoptimize depth as well as estimate pose by minimizing photometric loss.

![Untitled](https://user-images.githubusercontent.com/106483656/186256100-ebd2317c-a72f-4a27-a6d0-f08a90ecd4aa.jpg)

**Inference with pose hints:**

![Untitledd](https://user-images.githubusercontent.com/106483656/186257955-6ad1c0a2-fb80-4474-8681-d1a454154a6d.jpg)


4.

5. They train the network on KITTI Depth Prediction Dataset and evaluate the performance of their proposed method and their baseline method on KITTI Visual Odometry Dataset and KITTI Depth Prediction Dataset.

6. No it’s not.

7. **Results:**
1) When directly integrating IMU' to obtain pose, there is a large cumulative error 2) Their method outperforms monocular Libviso2 algorithm 3) When compared with their baseline method, their Pose Hints method can predict smoother and more accurate poses.
