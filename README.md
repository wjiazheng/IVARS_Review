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
- **Automatic instrument segmentation in robot-assisted surgery using deep learning**  
  [2018] :star:[paper](https://ieeexplore.ieee.org/abstract/document/8614125) :sunny:[code](https://github.com/ternaus/robot-surgery-segmentation)
- **Towards Unsupervised Learning for Instrument Segmentation in Robotic Surgery with Cycle-Consistent Adversarial Networks**  
  [2020] :star:[paper](https://ieeexplore.ieee.org/document/9340816) 
- **One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video**  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9561690)
- **TraSeTR: Track-to-Segment Transformer with Contrastive Query for Instance-level Instrument Segmentation in Robotic Surgery**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9811873)
- **Pseudo-label Guided Cross-video Pixel Contrast for Robotic Surgical Scene Segmentation with Limited Annotations**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9981798) :sunny:[code](https://github.com/yangyu-cuhk/PGV-CL)
- **MATIS: MASKED-ATTENTION TRANSFORMERS FOR SURGICAL INSTRUMENT SEGMENTATION**  
  [2023] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10230819/citations#citations) :sunny:[code](https://github.com/BCV-Uniandes/MATIS)
- **Robotic Scene Segmentation with Memory Network for Runtime Surgical Context Inference**  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/10342013) :sunny:[code](https://github.com/UVA-DSA/Runtime_RobScene_Seg_2Context)
- **Text Promptable Surgical Instrument Segmentation with Vision-Language Models**  
  [2023] :star:[paper](https://arxiv.org/abs/2306.09244) :sunny:[code](https://github.com/franciszzj/TP-SIS)
- **Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video**  
  [2020] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_65) :sunny:[code](https://github.com/zxzhaoeric/Semi-InstruSeg/)
- **MSDE-Net: A Multi-Scale Dual-Encoding Network for Surgical Instrument Segmentation**  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10366786) 
- **Structural and Pixel Relation Modeling for Semisupervised Instrument Segmentation From Surgical Videos**  
  [2024] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10359452) 
- **TMF-Net: A Transformer-Based Multiscale Fusion Network for Surgical Instrument Segmentation From Endoscopic Images**  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/9975835) 
- **Branch Aggregation Attention Network for Robotic Surgical Instrument Segmentation**  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/10158746) :sunny:[code](https://github.com/SWT-1014/BAANet)
- **Exploring Intra- and Inter-Video Relation for Surgical Semantic Scene Segmentation**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9779714) :sunny:[code](https://github.com/YuemingJin/STswinCL)
- **LSKANet: Long Strip Kernel Attention Network for Robotic Surgical Scene Segmentation**  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10330108) :sunny:[code](https://github.com/YubinHan73/LSKANet)
- **MSDESIS: Multitask Stereo Disparity Estimation and Surgical Instrument Segmentation**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9791423) :sunny:[code](https://github.com/dimitrisPs/msdesis)
- **SSIS-Seg: Simulation-Supervised Image Synthesis for Surgical Instrument Segmentation**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9783104) 
- **SurgNet: Self-Supervised Pretraining With Semantic Consistency for Vessel and Instrument Segmentation in Surgical Images**  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10354412) 
- **Video-Instrument Synergistic Network for Referring Video Instrument Segmentation in Robotic Surgery**  
  [2024] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10595513) :sunny:[code](https://github.com/whq-xxh/RSVIS)
- **Surgical-DeSAM: decoupling SAM for instrument segmentation in robotic surgery**  
  [2024] :star:[paper](https://link.springer.com/article/10.1007/s11548-024-03163-6) :sunny:[code](https://github.com/YuyangSheng/Surgical-DeSAM)
- **Anchor-guided online meta adaptation for fast one-Shot instrument segmentation from robotic surgical videos**  
  [2021] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841521002851#:~:text=In%20this%20paper%2C%20we%20study%20the%20challenging%20one-shot,accessible%20source%29%20can%20adapt%20to%20the%20target%20instruments.) 
- **FUN-SIS: A Fully UNsupervised approach for Surgical Instrument Segmentation**  
  [2023] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000129#:~:text=Propose%20a%20generative-adversarial%20approach%20for%20unsupervised%20surgical%20tool,the%20noise%20properties%20of%20these%20motion-derived%20segmentation%20masks.)
- **LACOSTE: Exploiting stereo and temporal contexts for surgical instrument segmentation**  
  [2025] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841524003128#:~:text=In%20this%20work%2C%20we%20propose%20a%20novel%20LACOSTEmodel,and%20TEmporal%20images%20for%20improved%20surgical%20instrument%20segmentation.)
- **Reducing annotating load: Active learning with synthetic images in surgical instrument**  
  [2024] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841524001713#:~:text=Motivated%20by%20alleviating%20the%20experts%E2%80%99%20workload%20of%20annotating,for%20labeled%20real%20images%20while%20having%20comparable%20performance.) :sunny:[code](https://github.com/HaonanPeng/active_syn_generator)
- **SurgiNet: Pyramid Attention Aggregation and Class-wise Self-Distillation for Surgical Instrument Segmentation**  
  [2022] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841521003558)
- **Incorporating Temporal Prior from Motion Flow for Instrument Segmentation in Minimally Invasive Surgery Video**  
  [2019] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_49) :sunny:[code](https://github.com/keyuncheng/MF-TAPNet)
- **Prototypical Interaction Graph for Unsupervised Domain Adaptation in Surgical Instrument Segmentation**  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_26) :sunny:[code](https://github.com/CityU-AIM-Group/SePIG)
- **Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data**  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_39) :sunny:[code](https://github.com/tajwarabraraleef/coSegGAN)
- **Efficient Global-Local Memory for Real-Time Instrument Segmentation of Robotic Surgical Video**  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_33) :sunny:[code](https://github.com/jcwang123/DMNet)
- **Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need**  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_34) :sunny:[code](https://github.com/lofrienger/Single_SurgicalScene_For_Segmentation)
- **Surgical Scene Segmentation Using Semantic Image Synthesis with a Virtual Surgery Environment**  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_53) :sunny:[code](https://sisvse.github.io/)
- **AdaptiveSAM: Towards Efficient Tuning of SAM for Surgical Scene Segmentation**  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-66958-3_14) :sunny:[code](https://github.com/JayParanjape/biastuning)
- **SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation**  
  [2024] :star:[paper](https://dl.acm.org/doi/10.1609/aaai.v38i7.28514) :sunny:[code](https://github.com/wenxi-yue/SurgicalSAM)
- **ISINet: An Instance-Based Approach for Surgical Instrument Segmentation**  
  [2020] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_57) :sunny:[code](https://github.com/BCV-Uniandes/ISINet)
- **Min-Max Similarity: A Contrastive Semi-Supervised Deep Learning Network for Surgical Tools Segmen**  
  [2023] :star:[paper](https://arxiv.org/abs/2203.15177) :sunny:[code](https://github.com/AngeLouCN/Min_Max_Similarity)
- **Graph-Based Surgical Instrument Adaptive Segmentation via Domain-Common Knowledge**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9583929) :sunny:[code](https://github.com/CityU-AIM-Group/Prototypical-Graph-DA)
- **Endo-Sim2Real: Consistency learning-based domain adaptation for instrument segmentation**  
  [2020] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_75) 
- **SAM Meets Robotic Surgery: An Empirical Study in Robustness Perspective**  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_23) 
- **Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments**  
  [2020] :star:[paper](https://ieeexplore.ieee.org/document/9197425) 
- **Pyramid Attention Aggregation Network for Semantic Segmentation of Surgical Instruments**  
  [2020] :star:[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6850#:~:text=In%20this%20paper%2C%20a%20novel%20network%2C%20Pyramid%20Attention,to%20aggregate%20multi-scale%20attentive%20features%20for%20surgical%20instruments.)
- **Space Squeeze Reasoning and Low-Rank Bilinear Feature Fusion for Surgical Image Segmentation**  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9722971) 

## Surgical_Action_Triplet_Recognition 
- Automatic Gesture Recognition in Robot-assisted Surgery with Reinforcement Learning and Tree Search
- [] :star:[paper]() :sunny:[code]()
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










