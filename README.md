# Intraoperative Video Analysis in Robotic Surgery: A Literature Review 

Intraoperative video analysis has emerged as a pivotal technology in robot-assisted minimally invasive surgery, enabling real-time interpretation of laparoscopic visual streams to enhance surgical precision, operational safety, and postoperative recovery. 

We categorized intraoperative video analysis in robotic surgery into six core sub-tasks:  
*[Surgical Image Semantic Segmentation](#Surgical_Image_Semantic_Segmentation)*   
*[Surgical Action Triplet Recognition](#Surgical_Action_Triplet_Recognition)*  
*[Stereo Matching and 3D Reconstruction in Robotic Surgery](#Stereo_Matching_and_3D_Reconstruction_in_Robotic_Surgery)*   
*[Preoperative-to-intraoperative Image Registration](#Preoperative-to-intraoperative_Image_Registration)*  
*[Unsupervised Soft-tissue Tracking](#Unsupervised_Soft-tissue_Tracking)*  
*[Surgical Phase Recognition](#Surgical_Phase_Recognition)*


## Surgical_Image_Semantic_Segmentation  
- Automatic instrument segmentation in robot-assisted surgery using deep learning  
  [2018] :star:[paper](https://ieeexplore.ieee.org/abstract/document/8614125) :sunny:[code](https://github.com/ternaus/robot-surgery-segmentation)
- Towards Unsupervised Learning for Instrument Segmentation in Robotic Surgery with Cycle-Consistent Adversarial Networks
- One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video
- TraSeTR: Track-to-Segment Transformer with Contrastive Query for Instance-level Instrument Segmentation in Robotic Surgery
- Pseudo-label Guided Cross-video Pixel Contrast for Robotic Surgical Scene Segmentation with Limited Annotations
- MATIS: MASKED-ATTENTION TRANSFORMERS FOR SURGICAL INSTRUMENT SEGMENTATION
- Robotic Scene Segmentation with Memory Network for Runtime Surgical Context Inference
- Text Promptable Surgical Instrument Segmentation with Vision-Language Models
- Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video
- MSDE-Net: A Multi-Scale Dual-Encoding Network for Surgical Instrument Segmentation
- Structural and Pixel Relation Modeling for Semisupervised Instrument Segmentation From Surgical Videos
- TMF-Net: A Transformer-Based Multiscale Fusion Network for Surgical Instrument Segmentation From Endoscopic Images
- Branch Aggregation Attention Network for Robotic Surgical Instrument Segmentation
- Exploring Intra- and Inter-Video Relation for Surgical Semantic Scene Segmentation
- LSKANet: Long Strip Kernel Attention Network for Robotic Surgical Scene Segmentation
- MSDESIS: Multitask Stereo Disparity Estimation and Surgical Instrument Segmentation
- SSIS-Seg: Simulation-Supervised Image Synthesis for Surgical Instrument Segmentation
- SurgNet: Self-Supervised Pretraining With Semantic Consistency for Vessel and Instrument Segmentation in Surgical Images
- Video-Instrument Synergistic Network for Referring Video Instrument Segmentation in Robotic Surgery
- Surgical-DeSAM: decoupling SAM for instrument segmentation in robotic surgery
- Anchor-guided online meta adaptation for fast one-Shot instrument segmentation from robotic surgical videos
- FUN-SIS: A Fully UNsupervised approach for Surgical Instrument Segmentation
- LACOSTE: Exploiting stereo and temporal contexts for surgical instrument segmentation
- Reducing annotating load: Active learning with synthetic images in surgical instrument
- SurgiNet: Pyramid Attention Aggregation and Class-wise Self-Distillation for Surgical Instrument Segmentation
- Incorporating Temporal Prior from Motion Flow for Instrument Segmentation in Minimally Invasive Surgery Video
- Prototypical Interaction Graph for Unsupervised Domain Adaptation in Surgical Instrument Segmentation
- Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data
- Efficient Global-Local Memory for Real-Time Instrument Segmentation of Robotic Surgical Video
- Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need
- Surgical Scene Segmentation Using Semantic Image Synthesis with a Virtual Surgery Environment
- AdaptiveSAM: Towards Efficient Tuning of SAM for Surgical Scene Segmentation
- SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation
- ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
- Min-Max Similarity: A Contrastive Semi-Supervised Deep Learning Network for Surgical Tools Segmen
- Graph-Based Surgical Instrument Adaptive Segmentation via Domain-Common Knowledge
- Endo-Sim2Real: Consistency learning-based domain adaptation for instrument segmentation
- SAM Meets Robotic Surgery: An Empirical Study in Robustness Perspective
- Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments
- Pyramid Attention Aggregation Network for Semantic Segmentation of Surgical Instruments
- Space Squeeze Reasoning and Low-Rank Bilinear Feature Fusion for Surgical Image Segmentation

## Surgical_Action_Triplet_Recognition 
- Automatic Gesture Recognition in Robot-assisted Surgery with Reinforcement Learning and Tree Search
- Multi-Task Recurrent Neural Network for Surgical Gesture Recognition and Progress Prediction
- Relational Graph Learning on Visual and Kinematics Embeddings for Accurate Gesture Recognition in Robotic Surgery
- Surgical Gesture Recognition Based on Bidirectional Multi-Layer Independently RNN with Explainable Spatial Feature Extraction
- Surgical Triplet Recognition via Diffusion Model
- MT-FiST: A Multi-Task Fine-Grained Spatial-Temporal Framework for Surgical Action Triplet Recognition
- Forest Graph Convolutional Network for Surgical Action Triplet Recognition in Endoscopic Videos
- Gesture Recognition in Robotic Surgery With Multimodal Attention
- Instrument-Tissue Interaction Detection Framework for Surgical Video Understanding
- CholecTriplet2021: A benchmark challenge for surgical action triplet recognition
- Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos
- Using 3D Convolutional Neural Networks to Learn Spatiotemporal Features for Automatic Surgical Gesture Recognition in Video
- Tail-Enhanced Representation Learning for Surgical Triplet Recognition
- Self-distillation for Surgical Action Recognition
- Surgical Action Triplet Detection by Mixed Supervised Learning of Instrument-Tissue Interactions
- Surgical Activity Triplet Recognition via Triplet Disentanglement
- Chain-of-Look Prompting for Verb-centric Surgical Triplet Recognition in Endoscopic Videos
- Concept Graph Neural Networks for  Surgical Video Understanding
- MT4MTL-KD: A Multi-Teacher Knowledge Distillation Framework for Triplet Recognition
- DATA SPLITS AND METRICS FOR METHOD BENCHMARKING ON SURGICAL ACTION TRIPLET DATASETS
- Parameter-efﬁcient framework for surgical action triplet recognition

## Stereo_Matching_and_3D_Reconstruction_in_Robotic_Surgery
- A Real-Time Interactive Augmented Reality Depth Estimation Technique for Surgical Robotics
- REAL-TIME COARSE-TO-FINE DEPTH ESTIMATION ON STEREO ENDOSCOPIC IMAGES WITH SELF-SUPERVISED LEARNING
- Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes
- Unsupervised-Learning-Based Continuous Depth and Motion Estimation With Monocular Endoscopy for Virtual Reality Minimally Invasive Surgery
- EndoMODE: A Multimodal Visual Feature-Based Ego-Motion Estimation Framework for Monocular Odometry and Depth Estimation in Various Endoscopic Scenes
- Bidirectional Semi-Supervised Dual-Branch CNN for Robust 3D Reconstruction of Stereo Endoscopic Images via Adaptive Cross and Parallel Supervisions.
- A Robust Edge-Preserving Stereo Matching Method for Laparoscopic Images
- EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos
- MonoPCC: Photometric-invariant cycle constraint for monocular depth estimation of endoscopic images
- Simultaneous Surgical Visibility Assessment, Restoration, and Augmented Stereo Surface Reconstruction for Robotic Prostatectomy
- EMDQ-SLAM: Real-Time High-Resolution Reconstruction of Soft Tissue Surface from Stereo Laparoscopy Videos
- Self-supervised Generative Adversarial Network for Depth Estimation in Laparoscopic Images
- EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera
- Enhanced Scale-Aware Depth Estimation for Monocular Endoscopic Scenes with Geometric Modeling
- Geometric Constraints for Self-supervised Monocular Depth Estimation on Laparoscopic Images with Dual-task Consistency
- Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery
- Self-supervised Depth Estimation in Laparoscopic Image Using 3D Geometric Consistency
- Bayesian Dense Inverse Searching Algorithm for Real-Time Stereo Matching in Minimally Invasive Surgery
- Deep Laparoscopic Stereo Matching with Transformers
- EndoSurf: Neural Surface Reconstruction of Deformable Tissues with Stereo Endoscope Videos
- Multi-view Guidance for Self-supervised Monocular Depth Estimation on Laparoscopic Images via Spatio-Temporal Correspondence
- Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers
- MSDESIS: Multitask Stereo Disparity Estimation and Surgical Instrument Segmentation
- E-DSSR: Efficient Dynamic Surgical Scene Reconstruction with Transformer-based Stereoscopic Depth Perception
- BDIS-SLAM: A lightweight CPU-based dense stereo SLAM for surgery
- Robust endoscopic image mosaicking via fusion of multimodal estimation
- Neural LerPlane Representations for Fast 4D Reconstruction of Deformable Tissues
- EndoGaussian: Real-time Gaussian Splatting for ynamic Endoscopic Scene Reconstruction
- StaSiS-Net: A stacked and siamese disparity estimation network for depth reconstruction in modern 3D laparoscopy

## Preoperative-to-intraoperative_Image_Registration
- Using Contours as Boundary Conditions for Elastic Registration during Minimally Invasive Hepatic Surgery
- Augmented Reality Navigation in Robot-Assisted Surgery with a Teleoperated Robotic Endoscope
- Augmented Reality Guided Laparoscopic Surgery of the Uterus
- Video-Based Soft Tissue Deformation Tracking for Laparoscopic Augmented Reality-Based Navigation in Kidney Surgery
- Point Cloud Registration in Laparoscopic Liver Surgery Using Keypoint Correspondence Registration Network
- Automatic preoperative 3d model registration in laparoscopic liver resection
- Automatic, global registration in laparoscopic liver surgery
- Learning feature descriptors for pre- and intra-operative point cloud matching for laparoscopic liver registration
- Automatic localization of endoscope in intraoperative CT image: A simple approach to augmented reality guidance in laparoscopic surgery
- The status of augmented reality in laparoscopic surgery as of 2016
- The value of Augmented Reality in surgery - A usability study on laparoscopic liver surgery
- An objective comparison of methods for augmented reality in laparoscopic liver resection by preoperative-to-intraoperative image fusion from the MICCAI2022 challenge
- Image-Based Incision Detection for Topological Intraoperative 3D Model Update in Augmented Reality Assisted Laparoscopic Surgery
- Using Multiple Images and Contours for Deformable 3D-2D Registration of a Preoperative CT in Laparoscopic Liver Surgery
- DNA-DIR: 2D-3D GEOMETRY EMBEDDING FOR INTRAOPERATIVE PARTIAL-TO-FULL REGISTRATION
- Using multiple images and contours for deformable 3D–2D registration of a preoperative CT in laparoscopic liver surgery
- Real-to-Sim Registration of Deformable Soft Tissue with Position-Based Dynamics for Surgical Robot Autonomy
- Feature-Guided Nonrigid 3-D Point Set Registration Framework for Image-Guided Liver Surgery: From Isotropic Positional Noise to Anisotropic Positional Noise
- An Optimal Control Problem for Elastic Registration and Force Estimation in Augmented Surgery

## Unsupervised_Soft-tissue_Tracking
- Ada-Tracker: Soft Tissue Tracking via Inter-Frame and Adaptive-Template Matching
- Surgical Tattoos in Infrared: A Dataset for Quantifying Tissue Tracking and Mapping
- Video-Based Soft Tissue Deformation Tracking for Laparoscopic Augmented Reality-Based Navigation in Kidney Surgery
- Evaluating Unsupervised Optical Flow for Keypoint Tracking in Laparoscopic Videos
- Tracker Learning Surgical Images By Self-Supervised Learning: An Enhanced Unsupervised Deep Tracking Approach

## Surgical_Phase_Recognition
- C-ECT: Online Surgical Phase Recognition with Cross-Enhancement Causal Transformer
- SKiT: a Fast Key Information Video Transformer for Online Surgical Phase Recognition
- EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos
- SV-RCNet: Workflow Recognition From Surgical Videos Using Recurrent Convolutional Network
- Temporal Memory Relation Network for Workflow Recognition From Surgical Video
- Exploring Segment-Level Semantics for Online Phase Recognition From Surgical Videos
- Cascade Multi-Level Transformer Network for Surgical Workflow Analysis
- Federated Cycling (FedCy): Semi-Supervised Federated Learning of Surgical Phases
- LAST: LAtent Space-Constrained Transformers for Automatic Surgical Phase Recognition and Tool Presence Detection
- Less Is More: Surgical Phase Recognition From Timestamp Supervision
- Semi-supervised learning with progressive unlabeled data excavation for label-efficient surgical workflow recognition
- LoViT: Long Video Transformer for surgical phase recognition
- Hard Frame Detection and Online Mapping for Surgical Phase Recognition
- OperA: Attention-Regularized Transformers for Surgical Phase Recognition
- Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer
- HecVL: Hierarchical Video-Language Pretraining for Zero-Shot Surgical Phase Recognition
- Label-Guided Teacher for Surgical Phase Recognition via Knowledge Distillation
- MuST: Multi-scale Transformers for Surgical Phase Recognition
- Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition
- Retrieval of Surgical Phase Transitions Using Reinforcement Learning
- Comparative validation of machine learning algorithms for surgical workflow and skill analysis with the HeiChole benchmark
- On the pitfalls of Batch Normalization for end-to-end video learning: A study on surgical workflow analysis
- Frequency-Based Temporal Analysis Network for Accurate Phase Recognition from Surgical Videos
- Intelligent surgical workflow recognition for endoscopic submucosal dissection with real-time animal study
- Dissecting Self-Supervised Learning Methods for Surgical Computer Vision
- Multi-task recurrent convolutional network with correlation loss for surgical video analysis










