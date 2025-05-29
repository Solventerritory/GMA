Computer Vision To Detect Abnormal Fidgety Movements in Infants

Abstract: - Early identification of neuromotor impairments in infants is critical for timely intervention but is currently reliant on expert visual assessments, which are often inaccessible in under-resourced settings. This motivates the need for automated, low-cost tools that leverage widely available data sources such as mobile-recorded videos. In this study, we attempted to extract and analyze movement kinematics from infant videos to detect deviations indicative of neuromotor risk. Our methodology involved keypoint tracking using multiple techniques including Lucas-Kanade optical flow, dense optical flow, and MediaPipe-based 3D motion capture. We extracted features such as velocity, acceleration, spatial position (X, Y), motion entropy, and applied statistical measures including Kullback-Leibler (KL) divergence and a Naïve Gaussian Bayesian Surprise metric to quantify deviations from healthy movement patterns. Despite promising directions, challenges included inconsistent keypoint detection, jittery tracking, and lack of sufficient labeled data to train or validate models effectively. Advanced methods such as TAPTR and 6D motion analysis were explored, but faced compatibility or dataset limitations. Our findings suggest that while current techniques show potential, robust tracking and geometric consistency remain major hurdles.

Introduction:
Developmental disorders, particularly those resulting from neuromotor impairments, represent a major cause of childhood disability, affecting an estimated 5–10% of children globally. In the United States alone, this corresponds to approximately 3.7 to 7.4 million children. These conditions often lead to lifelong limitations, making early detection and intervention essential. Research has shown that timely intervention during infancy can improve developmental outcomes. However, existing diagnostic techniques heavily depend on expert clinical assessment, which may not be widely available in low-resource settings. This underscores the urgent need for accessible, automated tools that can assess neuromotor risk in infants during the critical early months of life.
A reliable diagnostic test must offer both predictive power and accessibility. Many current assessments, such as the General Movements Assessment (GMA) and the Hammersmith Infant Neurological Examination (HINE), have demonstrated high sensitivity and specificity. Despite their effectiveness, these methods require trained professionals and controlled clinical environments, thus limiting scalability. Furthermore, even in well-resourced healthcare systems, the logistical demands of repeated expert assessments can hinder early and continuous monitoring.
To overcome these limitations, sensor-based methods—including wearable devices and 3D motion capture systems—have been developed for infant movement analysis. While such systems offer quantitative insights, they are often expensive, time-consuming, and impractical outside lab or hospital environments. As a more scalable alternative, recent research has turned to 2D video-based assessments, leveraging optical flow and pose estimation techniques to quantify movement features directly from video footage. These approaches are promising because they can be implemented on standard mobile devices, enabling at-home monitoring by caregivers. However, traditional optical flow methods often rely on centroid-based tracking and are limited in their ability to isolate fine-grained joint or limb movements. Many such systems also demand extensive manual tuning or rely on depth data and computationally heavy models trained on small datasets.
In this work, we explore several techniques to extract, track, and quantify infant movement from 2D videos using various motion estimation and pose-tracking algorithms. We aim to develop a pipeline that is not only accessible and low-cost, but also capable of capturing nuanced motion features indicative of neuromotor risk. Our methods include Lucas-Kanade and dense optical flow, 3D keypoint tracking with MediaPipe, ellipse fitting for Z-axis approximations, and temporal pose tracking using transformer-based architectures. We further attempt to quantify movement features like velocity, acceleration, and entropy, and evaluate deviation from normative motion data using statistical measures such as KL divergence and Bayesian Surprise.
Despite several challenges—including jittery keypoints, limited labeled datasets, and compatibility issues—we take important steps toward building a framework that combines classical computer vision with modern deep learning to assess infant movement in real-world conditions. Our goal is to pave the way for clinically relevant, fully automated neuromotor risk assessments using easily obtainable video data.

Methods:
Clinical Data
Infants aged 3–11 months were recruited from Ramiaya Hospital and the surrounding community. Full-term infants (>37 weeks gestation) with no significant medical conditions and preterm infants (<36 weeks gestation) from the NICU were included. Infants capable of walking were excluded. Informed consent was obtained, and ethical approval was granted by the hospital’s IRB.
A pediatric physical therapist assessed neuromotor risk using the Bayley Infant Neurodevelopmental Screener (BINS), classifying infants into low, moderate, or high-risk categories. Assessments were based on corrected age for preterm and chronological age for full-term infants.
Data collection took place in a sensor-equipped setup where infants lay supine on a 4×4 ft mat. Movements were recorded using high-resolution GoPro cameras. Sensorized toys were originally part of the setup, but for this analysis, only recordings without toys (to avoid occlusion) were used.

Pose Estimation and Keypoint Tracking
Optical Flow-Based Tracking
•	Method Used: Lucas-Kanade Optical Flow
•	Objective: Estimate motion trajectories of keypoints across consecutive frames to understand dynamic patterns.
•	Findings:
The approach struggled with tracking precision due to:
o	Low-contrast frames
o	Fine-grained and subtle infant movements
o	Potential occlusions and non-rigid body deformations
This highlighted the need for learning-based tracking methods with robustness to video quality and infant-specific motion characteristics.
________________________________________
Synthetic Data Augmentation for Pose Diversity
Generative Adversarial Networks (GANs)
•	Purpose: Address data scarcity and improve generalization by synthesizing realistic infant pose images.
•	Approach:
o	Initial GAN experiments aimed to replicate typical infant postures and variations.
o	Synthetic outputs were integrated into the training pipeline to diversify pose representations.
•	Ongoing Work:
Refinement of GAN architectures (e.g., StyleGAN, Pix2Pix) is planned to improve anatomical plausibility and image quality for training robustness.
________________________________________
Feature Extraction and Statistical Motion Profiling
Quantitative descriptors were derived from keypoint trajectories to support both unsupervised and supervised analysis:
•	Features Computed:
o	Velocity (Δx, Δy): Reflecting linear motion intensity
o	Acceleration: Highlighting abrupt changes
o	Entropy: Capturing randomness and variability of movement
o	KL Divergence: Measuring deviation between normal and abnormal motion distributions
•	Insight:
These metrics provide discriminative indicators of movement regularity and can serve as inputs to classifiers or anomaly detection systems.
3D Pose Estimation and Spatial Visualization
Mediapipe-Based Keypoint Extraction
•	Goal: Enhance analysis by including depth information and 3D joint positions.
•	Outcome:
o	Initial 3D plots were generated to visualize full-body motion patterns.
o	However, challenges remain in anatomical correctness and calibration, especially due to varying infant postures and occlusions.
________________________________________
Deep Learning for 6-DoF Motion Estimation
CNN-Based Joint and Orientation Prediction
•	Architecture: ResNet-based CNN trained to regress:
o	Joint Keypoints (JKPs) for position
o	Orientation Keypoints (OKPs) for rotational motion
•	Significance:
o	Provides a full 6-degree-of-freedom (3D position + 3D rotation) understanding of infant movements.
o	Enables detection of subtle neuromotor disorders by combining translational and rotational cues.
•	Advantages:
o	Deep residual connections in ResNet improved feature learning from occluded or complex infant poses.
o	The joint estimation of position and orientation is essential for comprehensive biomechanical assessments.
 

Results:-
Key Findings and Challenges:
•	Optical Flow Limitations: The Lucas-Kanade optical flow method struggled with accurate tracking due to low-contrast frames, subtle infant movements, and potential occlusions. This suggests that simpler optical flow techniques may not be robust enough for detailed analysis of infant fidgety movements.

•	Need for Learning-Based Methods: The difficulties with optical flow highlighted the necessity of employing learning-based tracking methods that can better handle variations in video quality and the complexities of infant motion.

•	Potential of Synthetic Data Augmentation: Initial experiments with GANs to generate synthetic infant pose images showed promise for addressing the issue of limited labeled data and improving the generalization of models. Ongoing work aims to refine GAN architectures for better anatomical plausibility and image quality.

•	Feature Extraction for Motion Profiling: Features like velocity, acceleration, entropy, and KL divergence were computed from keypoint trajectories. These metrics are intended to provide quantitative indicators of movement regularity and deviations that could be used for classification or anomaly detection. Image 1, showing the KL divergence between right elbow angles from two videos, visually demonstrates the application of this statistical measure to quantify differences in movement patterns. The high KL divergence (1.9423) suggests a substantial difference in the distribution of right elbow angles between the two videos.
 
•	Challenges in 3D Pose Estimation: While MediaPipe was used to generate initial 3D visualizations of full-body motion, anatomical correctness and calibration remained challenges, particularly due to varying infant postures and occlusions.

•	Promise of 6-DoF Motion Estimation: The development of a CNN-based architecture (ResNet) to predict both joint keypoints (JKPs) for position and orientation keypoints (OKPs) for rotational motion holds significant potential. This approach aims to provide a comprehensive 6-degree-of-freedom understanding of infant movements, potentially enabling the detection of subtle neuromotor disorders by analyzing both translational and rotational cues.

•	Temporal Analysis of Joint Angles: Image 2 illustrates the variation of left elbow, right elbow, left knee, and right knee angles over time (frame number). The plot shows dynamic changes in these joint angles, highlighting the complexity of infant movement and the need for temporal analysis techniques.
 

This image displays how the angles of specific joints (left elbow, right elbow, left knee, right knee) change over a sequence of frames in a video. The x-axis represents time (in frames), and the y-axis represents the angle in degrees. Each colored line tracks the angular movement of a particular joint throughout the observed period.


•	Hand Axis Analysis: Image 3 presents the major and minor axis lengths of ellipses of the left and right hands over time. This type of analysis could provide insights into the shape and size of the hands during movement, potentially revealing subtle differences in motor control.

 

This image shows the temporal evolution of the major and minor axis lengths of both the left and right hands. The top subplot corresponds to the left hand, and the bottom subplot corresponds to the right hand. Within each subplot, the blue line represents the major axis length, and the red line represents the minor axis length, both plotted against the frame number (time). This analysis aims to capture changes in the shape and potentially the orientation of the hands over the course of the video.

In summary, the results indicate that while there are promising avenues in applying computer vision techniques to analyze infant movements, significant challenges remain in achieving robust and accurate keypoint tracking and motion estimation. The use of statistical measures like KL divergence and advanced deep learning methods for 6-DoF motion analysis show potential for quantifying and detecting deviations in movement patterns. However, issues such as data scarcity, jittery tracking, and anatomical accuracy need to be addressed for the development of a reliable automated system for detecting abnormal fidgety movements in infants.



Discussion:-
This study explored the use of computer vision techniques to analyze infant movement from 2D videos and detect deviations indicative of neuromotor risk. We successfully implemented pose-estimation and keypoint tracking methods, including Lucas-Kanade optical flow, dense optical flow, and MediaPipe-based 3D motion capture, to extract kinematic features such as velocity, acceleration, and entropy. Our findings indicate that while these techniques hold promise for quantifying infant movement, several challenges remain in achieving robust and accurate motion estimation.
The Lucas-Kanade optical flow method encountered difficulties in accurately tracking keypoints due to low-contrast frames, subtle infant movements, and occlusions. This highlights a significant limitation of traditional optical flow techniques in the context of infant movement analysis, particularly for capturing the fine-grained details of fidgety movements. The observed limitations underscore the need for more sophisticated, learning-based tracking methods that can effectively handle the complexities and variability of infant motion in real-world video data.
To address the issue of limited labeled data, we explored the use of GANs for synthetic data augmentation. Initial experiments demonstrated the potential of GANs to generate realistic infant pose images, which could be used to diversify training data and improve the generalization of models. Ongoing work focuses on refining GAN architectures to enhance the anatomical plausibility and image quality of the synthetic data.
We computed several quantitative descriptors from the keypoint trajectories, including velocity, acceleration, entropy, and KL divergence. These metrics provide valuable information about movement regularity and deviations from typical patterns, and can potentially serve as inputs for automated systems designed to classify or detect abnormal movements. The successful application of KL divergence to quantify differences in movement patterns, as shown in Image 1, supports the utility of these statistical measures.
While MediaPipe facilitated the generation of initial 3D visualizations of full-body motion, we encountered challenges related to anatomical correctness and calibration. These challenges are likely attributable to the variability in infant postures and the presence of occlusions, which can hinder accurate 3D pose estimation.
The development of a CNN-based architecture (ResNet) for 6-DoF motion estimation represents a significant advancement in our ability to comprehensively understand infant movement. By predicting both joint keypoints (JKPs) and orientation keypoints (OKPs), this approach has the potential to capture subtle neuromotor disorders through the analysis of both translational and rotational cues. The ResNet architecture's deep residual connections likely contributed to improved feature learning from complex and partially occluded infant poses.
The temporal analysis of joint angles, as illustrated in Image 2, further emphasizes the complexity of infant movement and the need for advanced techniques that can capture dynamic changes in joint angles over time. Similarly, the hand axis analysis presented in Image 3 demonstrates a potential avenue for gaining insights into fine motor control and detecting subtle differences in hand movement patterns.
In conclusion, this study demonstrates the potential of computer vision techniques for quantifying infant movement and detecting deviations indicative of neuromotor risk. While significant challenges remain, particularly in achieving robust and accurate keypoint tracking, our exploration of advanced methods such as GANs and 6-DoF motion estimation offers promising directions for future research. Addressing the identified limitations, such as data scarcity, jittery tracking, and anatomical accuracy, is crucial for the development of a reliable automated system for detecting abnormal fidgety movements in infants.

Supplemental Information:-
Code and data referenced in the manuscript are provided at https://github.com/Solventerritory/GMA
