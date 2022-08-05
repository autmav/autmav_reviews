
# Object Detection and Tracking Literature Reviews

This is a set of brief reviews in the **Object Detection and Tracking** area on most recent research papers in the filed. 

Object tracking refers to the ability to estimate or predict the position of a target object in each consecutive frame in a video once the initial position of the target object is defined. On the other hand, object detection is the process of detecting a target object in an image or a single frame of the video.



## None-MDE-Aided Tracking (Control aspect)

Here, summarize the investigations in the new useful softwares which are included in an object detection and tracking project; Whether based on a motion model or visual data, or both.

We expect that most of the works listed in this part focus on the *guidance*, *navigation*, and *control* aspect of the problem in addition to the *computer vision* and *sensor fusion* algorithms.

After presenting the aforementioned data about each paper, answer to the following questions in your review:

1. Explain about the sensor fusion (if there is some)

2. Explain about the computer vision algorithm(s)

3. Explain about the navigation, guidance, and control strategy

4. Describe the performance in terms of technical parameters related to the tracking mission. 

5. Explain about the mission and the hardware


### Autonomous Human-Following Drone for Monitoring a Pedestrian from Constant Distance and Direction

[Link to the Paper](https://drive.google.com/file/d/1p6NsPXuzjZfd4ESUXRtnRvGcS1R0wn1F/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 
Hiroto Yamashita, Takashi Morimoto, Ikuhisa Mitsugami

**Date:** 
2021

**Journal or Conference:**
2021 IEEE 10th Global Conference on Consumer Electronics (GCCE)

#### Review:

Distance evaluation: They evaluated the length between the Neck and Hip points in the captured image while walking and standing. the length was kept almost constant except for moments when the person started walking or stopped.
Direction evaluation: The drone moved to the position behind the subject again, although there was some delay in the movement change of the subject.

#### Answers:

1. Not used.

2. CNN-based 2D pose estimation by OpenPose and 3D pose estimation by “Martinez et al” method.

3. The 3D joint points are estimated from 2D points with different skeleton model.
PID controller is applied. input values are the longitudinal, movement speed, vertical movement speed, and rotational, angular speed.
The human direction, Theta is calculated. This system controls the left-right movement speed of the drone so that theta approaches the target value.

4.  It is a real-time system that estimate humans 2D and 3D pose and Control parameter are calculated from skeleton points.
Experimental results confirmed that our system ran with reasonable response time and positional stability.

5. An autonomous drone system that follows a pedestrian from a certain distance and direction for human video analysis without environmental constraints. +Tello drone +RGB camera


### Multiple human tracking in drone image

[Link to the Paper](https://link.springer.com/article/10.1007/s11042-018-6141-z)

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:**

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.


## MDE for Object Detection and Tracking

The applications of MDE in the object detection and tracking must be summarized here

Try to answer the folloeing questions in your review:

1. What are the drone type and config?

2. What's the mission? Tracking a car? Human? Describe how hard the mission is and the mission constraints. How, and with what details, is the system implemented?

3. What are the sensors or pseudo sensors? Is the system V-INS? Just Visual? Or something else?

4. What data are fused?

5. What is the overal method to detect or track an object?

6. If there is, discuss more about the object motion model 


Add the papers you find in the following:

### Depth Estimation Matters Most: Improving Per-Object Depth Estimation for Monocular 3D Detection and Tracking

[Link to the Paper](https://arxiv.org/pdf/2206.03666.pdf)

[Link to the Source Code]()

**Authors:** Longlong Jing, Ruichi Yu, Henrik Kretzschmar, Kang Li, Charles R. Qi, Hang Zhao, Alper Ayvaci, Xu Chen, Dillon Cower, Yingwei Li, Yurong You, Han Deng, Congcong Li, and Dragomir Anguelov, Waymo LLC, Johns Hopkins University, Cornell University

**Date:**  8 Jun 2022

**Journal or Conference:**
IEEE

#### Review:

1)They conducted a systematic analysis identifying that per-object depth estimation is a major performance bottleneck of current 3D monocular detection and tracking-by-detection methods. 2) They proposed a novel method that fuses pseudo-LiDAR and RGB information across the temporal domain to significantly enhance per-object depth estimation performance. 3) They demonstrated that with the enhanced depth, the performance of monocular 3D detection and tracking can be significantly improved. 4) Future works can include end-to-end training of the proposed method.
Detectors such as RTM3D with the AB3D tracker, are state-of-the-art.

#### Answers:

1. Their full model Pseudo-LiDAR-RGB-Tracklet (PRT fusion) fuses information from both 2D and 3D representations across multiple frames. This fusion method achieves the state-of-the-art performance of per-object depth estimation on the Waymo Open Dataset, the KITTI detection dataset, and the KITTI MOT dataset. 

2. Fused feature is obtained using deep neural network to fuse the two features on PL(Pseudo-LiDAR representation for the object within the bounding box in the image plane) and the RGB image features. Also compact features for the entire image can be extracted by using a pre-trained convolution neural network.

3. Not mentioned.

4. The analysis suggests that depth has the most significant impact on detection (Average Precision) and tracking (MOTA) performance; with the stateof the-art monocular 3D detector CenterNet (ranked 1st on nuScene dataset) and the AB3D tracker. They use “+ GT” to indicate that they replaced the prediction with the ground truth.

5. The extraction process of the pseudo-LiDAR representation consists of three steps: (1) dense depth estimation for each image, (2) lifting predicted dense depth into pseudo-LiDAR, and (3) pseudo-LiDAR representation extraction with a neural network. For any RGB image, the depth estimation can be accomplished by using a dense depth estimation network.
Hardware details are not mentioned.

### Deep Learning for Real-Time 3D Multi-Object Detection, Localization, and Tracking: Application to Smart Mobility

[Link to the Paper](https://www.mdpi.com/1424-8220/20/2/532)

[Link to the Source Code]()

**Authors:** 
Antoine Mauri, Redouane Khemmar, Benoit Decoux, Nicolas Ragot, Romain Rossi, Rim Trabelsi, Rémi Boutteau , Jean-Yves Ertaud and Xavier Savatier
**Date:** 
Received: 19 November 2019; Accepted: 14 January 2020; Published: 18 January 2020
**Journal or Conference:**
MDPI journal

#### Review:

They improve SORT approach for 3D object tracking and introduce an extended Kalman filter to better estimate the position of objects. Extensive experiments carried out on KITTI dataset prove that their proposal outperforms state-of-the-art approches.

#### Answers:

1. **Hardware:** Nvidia GPU with at least 4 GB of VRAM is crucial. To perform the training of different deep learning architectures, the supercomputer MYRIA is utilized,
For testing, they use 2 servers equipped with 2 GPU GTX1080Ti each with 11 GB of VRAM and a computer with GPU GTX 1050 4 GB, 16 GB of RAM memory and CPU i5 8300 h.

2. An end-to-end deep learning based system for multi-object detection, depth estimation, localisation, and tracking for realistic road environments is presented.

3.**Sensors/Pseudo sensors:** Object tracking + depth Estimation for Localization + stereoscopic sensor + Intel RealsenseTM D435 cameras (Santa Clara, CA, USA)
*Just for idea* -> Visual SLAM sensor has the ability to provide reliable results when used indoors but has not been tested in outdoor conditions.

4. Not mentioned

5. They propose a detector-based on YOLOv3. Subsequently, to localize the detected objects, they put forward an adaptive method aiming to extract 3D information, i.e., depth maps. They use 2 approaches, Monodepth2 and MADNet to design the second module of object localisation. Finaly, a new object tracking method is introduced based on an improved version of the SORT approach. Extended Kalman Filter is presented to improve the estimation of object’s positions.
Unlike traditional tracking approaches which require target initialization beforehand, their approach consists of using information from object detection and distance estimation to initialize targets and to track them later

6. Not mentioned

7. **Neural Networks:** two approaches for second module of object localisation.-> Monodepth2 for monocular vision and MADNEt for stereoscopic vision (These approaches are then evaluated over KITTI datasets containing depth information)

8. **Environments:** They aim to use our solution in both indoor (such as smart wheelchair for disabled people) and outdoor (soft mobility for car and tramway) for road environments.

9. **Dataset:** due to the lack of a publicly available dataset for railway environments (barring RailSem19 dataset),they propose to extend the current version of their system by including a *new stereoscopic sensor* so that they can collect their own dataset under outdoor conditions with large and adjustable baselines.
Along with KITTI, many other datasets are available such as CityScapes, Pascal VO, MS-COCO, ImageNet and OpenImages devoted to the road domain

### Realtime Object-aware Monocular Depth Estimation in Onboard Systems


[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
Sangil Lee, Chungkeun Lee, Haram Kim, and H. Jin Kim*
**Date:** 
January 19, 2021
**Journal or Conference:**
Springer journal
#### Review:

Its performance is comparable to that of MDE algorithms which train depth indirectly (or directly) from stereo image pairs (or depth image), and better than that of algorithms trained with monocular images only, in terms of the error and the accuracy. Also their computational load is much lighter than the learning-based methods.

#### Answers:

1. Not mentioned

2. It proposes the object depth estimation in real-time for autonomous driving, using only a monocular camera with a geometric approach, in an onboard computer with a low-cost GPU from a sparse feature-based visual odometry algorithm. (The camera’s own 6-DoF pose should be recognized in advance.)
**Implementation details:** they implement simultaneous object detection(requires GPUintensive computation) and depth estimation(requires CPU-intensive computation) with dual-threading in an onboard environment with NVIDIA Jetson TX2 platform.
**Implementation parts:** 1) feature extraction/tracking and object detection from a single image frame. 2)feature’s depth estimation and object tracking/merging. 3) object depth estimation and upkeep step.
**Validation:** they validate the scene depth accuracy of sparse features with KITTI and its ground-truth depth map made from LiDAR observations quantitatively, and the depth of detected object with the Hyundai driving datasets and satellite maps qualitatively.
The object classification groups: road boundary lines, traffic light, traffic signs, and vehicles

3. **Pseudo sensors:** learning-based object detection, depth estimation

4. Data fusion is not mentioned

5. In the object recognition stage, objects are detected and tracked and their depths are calculated from feature points within their ROI. As the feature-based scene reconstruction algorithm shares KLT tracking results with the object recognition method, it is not necessary to perform tracking twice in the object tracking procedure, thus reducing execution time.

6. Not mentioned

7. **Future works:** next step is to estimate the motion of multiple static and dynamic objects in the scene and to compute their depths by compensating the ego-motion.

8. **Dataset:** KITTI, Waymo open dataset and nuScenes do not provide labels for static objects. So they utilize the Hyundai MnSoft driving dataset including many annotations in dicating static objects such as traffic signs and road signs.


### Monocular 3D Multi-Object Tracking with an EKF Approach for Long-Term Stable Tracks

[Link to the Paper](https://drive.google.com/file/d/1rJ3PnFiJbZcrKbh4VGLAxsz_62sTOKMX/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 
Andreas Reich, Hans-Joachim Wuensche

**Date:** 
Date of Conference: 01-04 November 2021

**Journal or Conference:**
IEEE journal
2021 IEEE 24th International Conference on Information Fusion

#### Review:

1) Accurate estimation of 3D state and its uncertainty by using 2D bounding boxes and thus avoiding temporal correlations of measurement errors. It's extensible by detections from other sensors such as LiDAR and radar. 2) Compared to state-of-the-art approaches (based on deep networks), this approach allows more insights and the output is more explainable.It explicitly separates geometric-based from appearance-based association. 3) With the proposed tracker they achieve state-of-the-art results in the KITTI multi-object tracking benchmark for both classes, cars and pedestrians. In terms of low numbers of identity switches and track fragmentations, they even outperform all other approaches, indicating long-term stable tracks.
Future works: adding measurements from LiDAR and radar sensors

#### Answers:

1- Drone details is not mentioned. 
**Hardware:** It is focused on cameras. The network requires approximately 80 ms to process one image on a regular consumer graphics card (Nvidia GeForce GTX 1060 of 2014).

2- It tracks cars and pedestrian.
To process all detections of one image for implementation in Python, their tracking needs 3 ms (EFK computational efficiency).
Other details are reported in the review part.

3- Type: Visual based. All images and sensor data are from KITTI dataset.
**Pseudo sensors:** Extended Kalman filter based monocular 3D multi-object tracker and detecter and depth estimation
**Details:** For closer objects the projected center is more difficult to determine. So they increase the noise of them.
They pre-trained the network on the NuScenes dataset and fine-tuned it on the training split of the KITTI dataset. They configured the network to output 3D detections besides the 2D ones in order to use this net also for track initialization. In comparison to other monocular 3D object detectors, the former has a very low FP and FN rate.

4- The EKF(Extended Kalman filter) accurately estimates the real state uncertainties and can be therefore used as a starting point for multi-sensor fusion.
**Performance:** They evaluated EKF with y_2D and y_3D in the measurement update. The former has a far better performance. In the latter case the uncertainty gets too small and is useless for fusion with detections from other sensors.

5- They achieved state-ofthe-art results (on the KITTI dataset with an association solely based on 2D bounding box comparison), with very robust tracks in terms of the HOTA score. Their approach can use detections from an arbitrary monocular 3D object detector. Therefore it can even improve its performance with better detections.
To improve tracking performance they tested an Unscented Kalman filter (UKF), which is known to handle non-linear measurement equations better. However, the UKF is still unimodal and does not provide better scores. Multimodal estimators such as the Particle filter or a variant of a Gaussian Mixture filter would be more appropriate. But they do not consider these types of Bayes filters for simplicity.

6- Not mentioned.

7- **Challenges:** Challenging scenarios for their tracker are bad aspect angle estimates at and shortly after track initialization. This especially happens for distant objects at crossings and it can be handled by including detections from other sensors.

### Real-time Monocular 3D People Localization and Tracking on Embedded System

[Link to the Paper](https://drive.google.com/file/d/1HKHRA1zM6AxbsaZgxfexzKUygNpzXZTo/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 
Yipeng Zhu, Tao Wang, Shiqiang Zhu

**Date:** 
July 3-5, 2021

**Journal or Conference:**
2021 6th IEEE International Conference on Advanced Robotics and Mechatronics

#### Review:
1) A real-time monocular 3D people localization method is proposed, with efficient monocular depth estimation and 2D object detection neural networks adopted.
2) An evaluation index is proposed to reflect the monocular depth estimation quality. A Kalman filter based tracking module is adopted to improve localization accuracy.
3) The 3D people localization and tracking pipeline is implemented on an edge device and evaluated for error and speed. This lightweight implementation provides a cost-effective people localization solution for those applications with limited computing resource.

#### Answers:

1. **Hardware:** CSI camera which provides 30 fps RGB images of resolution 320x180 is used. MDE and 2D object detection module run on the Jetson Xavier NX developer kit. Drone model is not mentioned.

2. It proposed a lightweight monocular 3D people localization pipeline on the results of monocular depth estimation and 2D people detection with neural networks. Aiming for real-time performance on mobile devices, a fine-tuning friendly solution is implemented (in indoor environment) on an edge device and evaluated for error and speed. The overall performance reaches 12 fps with an acceptable accuracy compared to ground truth.

3. Passive vision-based localization

4. Temporal information

5. 2D object detection results are adopted for finding accurate people location. Finally, a Kalman filter based tracking module is adopted is adopted to fuse temporal information and improve the accuracy. It uses interpretable pipelines method for people localization. Temporal information is used to improve localization accuracy by a filter based tracking module. The system strikes a balance between high accuracy and high frame rate.

6. Not mentioned.

7. **Neural network:** MonoDepth2 , self-supervised monocular depth estimation method, has a satisfying performance with acceptable model complexity. It consists of a general U-net encoder-decoder structure.

### Design of a Robust System Architecture for Tracking Vehicle on Highway Based on Monocular Camera

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 

**Date:** 

**Journal or Conference:**

#### Review:

Type a paragraph

#### Answers:

1.

2.

3.

4.

5.

6.


### YOLO MDE: Object Detection with Monocular Depth Estimation

[Link to the Paper]()

[Link to the Source Code]()

**Authors:** 
Jongsub Yu and Hyukdoo Choi
**Date:** 
Published: 27 December 2021
**Journal or Conference:** MDPI journals

#### Review:

All details is mentioned below.

#### Answers:

1. A single Nvidia GeForce RTX 3090 (24 GB) + a single GPU of Tesla V100

2. It tracks cars and pedestrians.
Evaluation Metrics: error rate of the depth per object.
As experiment, they tried using two different network architectures: YOLO v3 and YOLO v4. Since we focused on higher detection speed, we used only one-stage detectors for the experiment.

3. **Sensors/pseudo sensors:** An 2D object detector with depth estimation using monocular camera images. (Just visual)

4. It doesn't use fusion.

5. This network architecture is based on YOLO v4, which is a fast and accurate one-stage object detector. They added *"only a single"* additional channel(unlike Recent 3D object detectors) to the output layer for (straightforward) depth estimation. To train depth prediction, they extract the closest depth from the 3D bounding box coordinates of ground truth labels in the KITTI dataset. They designed a novel loss function to train depth estimation striking a balance between near and far distance accuracy.

6. Not mentioned

7. **Performance:** This model achieved an AP of 71.68% for cars and 62.12% for pedestrians and a mean error rate of 3.71% in the KITTI 3D object detection dataset. It also achieved a detection speed of 25 FPS. This detection has far better performance and faster detection speed than the latest 3D detection models (e.g. D4LCN and GM3D).

8. **Further research:** this model may be applied to object tracking tasks. To improve the detection performance, adapting different object detection architecture such as EfficientDet is expected in future works.
