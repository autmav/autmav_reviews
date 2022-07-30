
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

**Journal or Conference:** ...
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

**Journal or Conference:** ...

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

**Journal or Conference:** ...
IEEE

#### Review:

1) They conducted a systematic analysis identifying that per-object depth estimation is a major performance bottleneck of current 3D monocular detection and tracking-by-detection methods. 2) They proposed a novel method that fuses pseudo-LiDAR and RGB information across the temporal domain to significantly enhance per-object depth estimation performance. 3) They demonstrated that with the enhanced depth, the performance of monocular 3D detection and tracking can be significantly improved. 4) Future works can include end-to-end training of the proposed method.
Detectors such as RTM3D with the AB3D tracker, are state-of-the-art.

#### Answers:

1. Their full model Pseudo-LiDAR-RGB-Tracklet (PRT fusion) fuses information from both 2D and 3D representations across multiple frames. This fusion method achieves the state-of-the-art performance of per-object depth estimation on the Waymo Open Dataset, the KITTI detection dataset, and the KITTI MOT dataset. 

2. Fused feature is obtained using deep neural network to fuse the two features on PL(Pseudo-LiDAR representation for the object within the bounding box in the image plane) and the RGB image features. Also compact features for the entire image can be extracted by using a pre-trained convolution neural network.

3. Not mentioned.

4. The analysis suggests that depth has the most significant impact on detection (Average Precision) and tracking (MOTA) performance; with the stateof the-art monocular 3D detector CenterNet (ranked 1st on nuScene dataset) and the AB3D tracker. They use “+ GT” to indicate that they replaced the prediction with the ground truth.

5. The extraction process of the pseudo-LiDAR representation consists of three steps: (1) dense depth estimation for each image, (2) lifting predicted dense depth into pseudo-LiDAR, and (3) pseudo-LiDAR representation extraction with a neural network. For any RGB image, the depth estimation can be accomplished by using a dense depth estimation network.
Hardware details are not mentioned.

### Deep Learning for Real-Time 3D Multi-Object Detection, Localisation, and Tracking: Application to Smart Mobility

[Link to the Paper](https://www.mdpi.com/1424-8220/20/2/532)

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



### Realtime Object-aware Monocular Depth Estimation in Onboard Systems


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


### Monocular 3D Multi-Object Tracking with an EKF Approach for Long-Term Stable Tracks

[Link to the Paper](https://drive.google.com/file/d/1rJ3PnFiJbZcrKbh4VGLAxsz_62sTOKMX/view?usp=sharing)

[Link to the Source Code]()

**Authors:** 
Andreas Reich, Hans-Joachim Wuensche

**Date:** 
Date of Conference: 01-04 November 2021

**Journal or Conference:** ...
IEEE journal
2021 IEEE 24th International Conference on Information Fusion (FUSION)

#### Review:

1) Accurate estimation of 3D state and its uncertainty by using 2D bounding boxes and thus avoiding temporal correlations of measurement errors. It's extensible by detections from other sensors such as LiDAR and radar. 2) Compared to state-of-the-art approaches (based on deep networks), this approach allows more insights and the output is more explainable.It explicitly separates geometric-based from appearance-based association. 3) With the proposed tracker they achieve state-of-the-art results in the KITTI multi-object tracking benchmark for both classes, cars and pedestrians. In terms of low numbers of identity switches and track fragmentations, they even outperform all other approaches, indicating long-term stable tracks.
Future works: adding measurements from LiDAR and radar sensors

#### Answers:

1. The EKF(Extended Kalman filter) accurately estimates the real state uncertainties and can be therefore used as a starting point for multi-sensor fusion.
They evaluated EKF with y_2D and y_3D in the measurement update. The former has a far better performance. Furthermore, in the latter case the uncertainty gets too small and is useless for fusion with detections from other sensors.

2. For closer objects the projected center is more difficult to determine. So they increase the noise of them.
They pre-trained the network on the NuScenes dataset and fine-tuned it on the training split of the KITTI dataset. They configured the network to output 3D detections besides the 2D ones in order to use this net also for track initialization. In comparison to other monocular 3D object detectors, the former has a very low FP and FN rate.

3. They present a multi object tracking approach composed of an Extended Kalman filter estimating the 3D state by using these detections for track initialization. They use state uncertainties transformed into the measurement space while completely ignoring appearance features.
To improve tracking performance they tested an Unscented Kalman filter (UKF), which is known to handle non-linear measurement equations better. However, the UKF is still unimodal and does not provide better scores. Multimodal estimators such as the Particle filter or a variant of a Gaussian Mixture filter would be more appropriate. But they do not consider these types of Bayes filters for simplicity.

4. They achieved state-ofthe-art results (on the KITTI dataset with an association solely based on 2D bounding box comparison), with very robust tracks in terms of the HOTA score. Their approach can use detections from an arbitrary monocular 3D object detector. Therefore it can even improve its performance with better detections.
To process all detections of one image for implementation in Python, their tracking needs 3 ms (EFK computational efficiency).

5. It is focused on cameras. The network requires approximately 80 ms to process one image on a regular consumer graphics card (Nvidia GeForce GTX 1060 of 2014). 

6. Challenging scenarios for their tracker are bad aspect angle estimates at and shortly after track initialization. This especially happens for distant objects at crossings and it can be handled by including detections from other sensors.

### Real-time Monocular 3D People Localization and Tracking on Embedded System

[Link to the Paper](https://drive.google.com/file/d/1HKHRA1zM6AxbsaZgxfexzKUygNpzXZTo/view?usp=sharing)

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


### Design of a Robust System Architecture for Tracking Vehicle on Highway Based on Monocular Camera

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


### YOLO MDE: Object Detection with Monocular Depth Estimation

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
