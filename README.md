# Intraoperative Video Analysis in Robotic Surgery: A Literature Review 

Intraoperative video analysis has emerged as a pivotal technology in robot-assisted minimally invasive surgery, enabling real-time interpretation of laparoscopic visual streams to enhance surgical precision, operational safety, and postoperative recovery. 

In this review, intraoperative video analysis for robotic surgery is categorized into six core sub-tasks, which can be grouped into three levels of semantic understanding in the surgical scene:

Pixel-level Intraoperative Video Analysis:

*[Surgical Image Semantic Segmentation](#Surgical_Image_Semantic_Segmentation)*   
*[Unsupervised Soft-tissue Tracking](#Unsupervised_Soft-tissue_Tracking)*  

Global-level Intraoperative Video Analysis:

*[Surgical Action Triplet Recognition](#Surgical_Action_Triplet_Recognition)*  
*[Surgical Phase Recognition](#Surgical_Phase_Recognition)*

Spatial-level Intraoperative Video Analysis:

*[Intraoperative Depth Estimation and 3D Reconstruction](#Intraoperative_Depth_Estimation_and_3D_Reconstruction)*   
*[Preoperative-to-intraoperative Image Registration](#Preoperative-to-intraoperative_Registration)*  

![image](https://github.com/wjiazheng/IVARS_Review/blob/main/fig1.png)

![image](https://github.com/wjiazheng/IVARS_Review/blob/main/fig3.png)
Statistical data of publications by journals and conferences in intraoperative video analysis over the last decade (2016-2025).

The reviewed papers are listed as follow:

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
- **【Endovis2017】2017 robotic instrument segmentation challenge**  
  [2019] :star:[paper](https://arxiv.org/abs/1902.06426) :triangular_flag_on_post:[dataset](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
- **【Endovis2018】2018 robotic scene segmentation challenge**  
  [2020] :star:[paper](https://arxiv.org/abs/2001.11190) :triangular_flag_on_post:[dataset](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/)

## Surgical_Action_Triplet_Recognition 
- Automatic Gesture Recognition in Robot-assisted Surgery with Reinforcement Learning and Tree Search  
  [2020] :star:[paper](https://ieeexplore.ieee.org/document/9196674) 
- Multi-Task Recurrent Neural Network for Surgical Gesture Recognition and Progress Prediction  
  [2020] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9197301)
- Relational Graph Learning on Visual and Kinematics Embeddings for Accurate Gesture Recognition in Robotic Surgery  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9561028) :sunny:[code](https://www.cse.cuhk.edu.hk/~yhlong/mrgnet.html)
- Surgical Gesture Recognition Based on Bidirectional Multi-Layer Independently RNN with Explainable Spatial Feature Extraction  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9561803) 
- Surgical Triplet Recognition via Diffusion Model  
  [2024] :star:[paper](https://arxiv.org/abs/2406.13210) 
- MT-FiST: A Multi-Task Fine-Grained Spatial-Temporal Framework for Surgical Action Triplet Recognition  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/10195982) :sunny:[code](https://github.com/Lycus99/MT-FiST)
- Forest Graph Convolutional Network for Surgical Action Triplet Recognition in Endoscopic Videos  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9831997) 
- Gesture Recognition in Robotic Surgery With Multimodal Attention  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9701436) :triangular_flag_on_post:[dataset](https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-data-server)
- Instrument-Tissue Interaction Detection Framework for Surgical Video Understanding  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10478628) :sunny:[code](https://gaiakoen.github.io/yanhu/research/Surgical_Scenarios_)
- Using 3D Convolutional Neural Networks to Learn Spatiotemporal Features for Automatic Surgical Gesture Recognition in Video  
  [2019] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_52) :sunny:[code](https://gitlab.com/nct_tso_public/surgical_gesture_recognition)
- Tail-Enhanced Representation Learning for Surgical Triplet Recognition  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_64) :sunny:[code](https://github.com/CIAM-Group/ComputerVision_Codes/tree/main/TERL)
- Self-distillation for Surgical Action Recognition  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_61) :sunny:[code](https://github.com/IMSY-DKFZ/self-distilled-swin)
- Surgical Action Triplet Detection by Mixed Supervised Learning of Instrument-Tissue Interactions  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_48) :sunny:[code](https://github.com/CAMMA-public/mcit-ig)
- Surgical Activity Triplet Recognition via Triplet Disentanglement  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_43) 
- Chain-of-Look Prompting for Verb-centric Surgical Triplet Recognition in Endoscopic Videos  
  [2023] :star:[paper](https://dl.acm.org/doi/10.1145/3581783.3611898) :sunny:[code](https://github.com/southnx/CoLSurgical)
- Concept Graph Neural Networks for  Surgical Video Understanding  
  [2024] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10195995) 
- MT4MTL-KD: A Multi-Teacher Knowledge Distillation Framework for Triplet Recognition  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10368037) :sunny:[code](https://github.com/CIAM-Group/ComputerVision_Codes/tree/main/)
- DATA SPLITS AND METRICS FOR METHOD BENCHMARKING ON SURGICAL ACTION TRIPLET DATASETS  
  [2022] :star:[paper](https://arxiv.org/abs/2204.05235) :sunny:[code](https://github.com/CAMMA-public) :triangular_flag_on_post:[dataset](http://camma.u-strasbg.fr/datasets)
- Parameter-efﬁcient framework for surgical action triplet recognition  
  [2024] :star:[paper](https://link.springer.com/article/10.1007/s11548-024-03147-6) :sunny:[code](https://github.com/Lycus99/LAM)
- 【JIGSAWS】A Dataset and Benchmarks for Segmentation and Recognition of Gestures in Robotic Surgery  
  [2017] :star:[paper](https://ieeexplore.ieee.org/document/7805258) :triangular_flag_on_post:[dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
- 【CholecT40】Recognition of instrument-tissue interactions in endoscopic videos via action triplets.  
  [2020] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_35) :triangular_flag_on_post:[dataset](https://github.com/CAMMA-public/tripnet)
- 【CholecT45】CholecTriplet2021: A benchmark challenge for surgical action triplet recognition  
  [2023] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000646) :sunny:[code](https://github.com/CAMMA-public/cholectriplet2021) :triangular_flag_on_post:[dataset](https://github.com/CAMMA-public/cholect45)
- 【CholecT50】Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos  
  [2022] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841522000846#:~:text=Out%20of%20all%20existing%20frameworks%20for%20surgical) :sunny:[code](https://github.com/CAMMA-public/rendezvous) :triangular_flag_on_post:[dataset](https://github.com/CAMMA-public/cholect50)
  
## Intraoperative_Depth_Estimation_and_3D_Reconstruction
- A Real-Time Interactive Augmented Reality Depth Estimation Technique for Surgical Robotics  
  [2019] :star:[paper](https://ieeexplore.ieee.org/document/8793610) 
- REAL-TIME COARSE-TO-FINE DEPTH ESTIMATION ON STEREO ENDOSCOPIC IMAGES WITH SELF-SUPERVISED LEARNING  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9434058) 
- Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9561508) 
- Unsupervised-Learning-Based Continuous Depth and Motion Estimation With Monocular Endoscopy for Virtual Reality Minimally Invasive Surgery  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9145848) 
- EndoMODE: A Multimodal Visual Feature-Based Ego-Motion Estimation Framework for Monocular Odometry and Depth Estimation in Various Endoscopic Scenes  
  [2025] :star:[paper](https://ieeexplore.ieee.org/document/10979973) 
- Bidirectional Semi-Supervised Dual-Branch CNN for Robust 3D Reconstruction of Stereo Endoscopic Images via Adaptive Cross and Parallel Supervisions.  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/10136208) 
- A Robust Edge-Preserving Stereo Matching Method for Laparoscopic Images  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9695464) 
- EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos  
  [2021] :star:[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841521001043) :sunny:[code](https://github.com/CapsuleEndoscope/EndoSLAM) :triangular_flag_on_post:[dataset](https://github.com/CapsuleEndoscope/EndoSLAM)
- MonoPCC: Photometric-invariant cycle constraint for monocular depth estimation of endoscopic images  
  [2025] :star:[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841525000817) :sunny:[code](https://github.com/adam99goat/MonoPCC)
- Simultaneous Surgical Visibility Assessment, Restoration, and Augmented Stereo Surface Reconstruction for Robotic Prostatectomy  
  [2018] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-00937-3_37) 
- EMDQ-SLAM: Real-Time High-Resolution Reconstruction of Soft Tissue Surface from Stereo Laparoscopy Videos  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_32) 
- Self-supervised Generative Adversarial Network for Depth Estimation in Laparoscopic Images  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_22) 
- EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_20) :sunny:[code](https://github.com/BeileiCui/EndoDAC)
- Enhanced Scale-Aware Depth Estimation for Monocular Endoscopic Scenes with Geometric Modeling  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_25) :sunny:[code](https://github.com/med-air/MonoEndoDepth)
- Geometric Constraints for Self-supervised Monocular Depth Estimation on Laparoscopic Images with Dual-task Consistency  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_45) :sunny:[code](https://github.com/MoriLabNU/GCDepthL)
- Self-supervised Depth Estimation in Laparoscopic Image Using 3D Geometric Consistency  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_2) 
- Bayesian Dense Inverse Searching Algorithm for Real-Time Stereo Matching in Minimally Invasive Surgery  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_32) :sunny:[code](https://github.com/JingweiSong/BDIS.git)
- Deep Laparoscopic Stereo Matching with Transformers  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_44) 
- EndoSurf: Neural Surface Reconstruction of Deformable Tissues with Stereo Endoscope Videos  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_2) :sunny:[code](https://github.com/Ruyi-Zha/endosurf.git)
- Multi-view Guidance for Self-supervised Monocular Depth Estimation on Laparoscopic Images via Spatio-Temporal Correspondence  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_41) :sunny:[code](https://github.com/MoriLabNU/MGMDepthL)
- Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers  
  [2021] :star:[paper](https://ieeexplore.ieee.org/document/9711118) :sunny:[code](https://github.com/mli0603/stereo-transformer)
- MSDESIS: Multitask Stereo Disparity Estimation and Surgical Instrument Segmentation  
  [2022] :star:[paper](https://ieeexplore.ieee.org/document/9791423) :sunny:[code](https://github.com/dimitrisPs/msdesis)
- E-DSSR: Efficient Dynamic Surgical Scene Reconstruction with Transformer-based Stereoscopic Depth Perception  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_40) 
- BDIS-SLAM: A lightweight CPU-based dense stereo SLAM for surgery  
  [2024] :star:[paper](https://link.springer.com/article/10.1007/s11548-023-03055-1) :sunny:[code](https://github.com/JingweiSong/BDIS-SLAM)
- Robust endoscopic image mosaicking via fusion of multimodal estimation  
  [2023] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841522003371) 
- Neural LerPlane Representations for Fast 4D Reconstruction of Deformable Tissues  
  [2023] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_5) :sunny:[code](https://github.com/Loping151/ForPlane)
- EndoGaussian: Real-time Gaussian Splatting for dynamic Endoscopic Scene Reconstruction  
  [2024] :star:[paper](https://arxiv.org/abs/2401.12561) :sunny:[code](https://yifliu3.github.io/EndoGaussian/)
- StaSiS-Net: A stacked and siamese disparity estimation network for depth reconstruction in modern 3D laparoscopy  
  [2022] :star:[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522000329)
- 【EndoNeRF】Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_41) :sunny:[code](https://github.com/med-air/EndoNeRF) :triangular_flag_on_post:[dataset](https://github.com/med-air/EndoNeRF)
- 【SCARED】Stereo correspondence and reconstruction of endoscopic data challenge  
  [2021] :star:[paper](https://arxiv.org/abs/2101.01133) :triangular_flag_on_post:[dataset](https://endovissub2019-scared.grand-challenge.org/)
  

## Preoperative-to-intraoperative_Registration
- Using Contours as Boundary Conditions for Elastic Registration during Minimally Invasive Hepatic Surgery  
  [2016] :star:[paper](https://ieeexplore.ieee.org/document/7759099) 
- Augmented Reality Navigation in Robot-Assisted Surgery with a Teleoperated Robotic Endoscope  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/10342282) :sunny:[code](https://github.com/moveit/moveit)
- Augmented Reality Guided Laparoscopic Surgery of the Uterus  
  [2020] :star:[paper](https://ieeexplore.ieee.org/document/9207920) 
- Video-Based Soft Tissue Deformation Tracking for Laparoscopic Augmented Reality-Based Navigation in Kidney Surgery  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10555430) 
- Point Cloud Registration in Laparoscopic Liver Surgery Using Keypoint Correspondence Registration Network  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10672536) 
- Automatic preoperative 3d model registration in laparoscopic liver resection  
  [2022] :star:[paper](https://link.springer.com/article/10.1007/s11548-022-02641-z) 
- Automatic, global registration in laparoscopic liver surgery  
  [2022] :star:[paper](https://link.springer.com/article/10.1007/s11548-021-02518-7) 
- Learning feature descriptors for pre- and intra-operative point cloud matching for laparoscopic liver registration  
  [2023] :star:[paper](https://link.springer.com/article/10.1007/s11548-023-02893-3) :sunny:[code](https://github.com/zixinyang9109/LiverMatch) :triangular_flag_on_post:[dataset](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
- Automatic localization of endoscope in intraoperative CT image: A simple approach to augmented reality guidance in laparoscopic surgery  
  [2016] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841516000153) 
- The status of augmented reality in laparoscopic surgery as of 2016  
  [2017] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841517300178) 
- The value of Augmented Reality in surgery - A usability study on laparoscopic liver surgery  
  [2023] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841523002037) 
- Image-Based Incision Detection for Topological Intraoperative 3D Model Update in Augmented Reality Assisted Laparoscopic Surgery  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_62) 
- DNA-DIR: 2D-3D GEOMETRY EMBEDDING FOR INTRAOPERATIVE PARTIAL-TO-FULL REGISTRATION  
  [2024] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10635158) 
- Using multiple images and contours for deformable 3D–2D registration of a preoperative CT in laparoscopic liver surgery  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_63) 
- Using Multiple Images and Contours for Deformable 3D-2D Registration of a Preoperative CT in Laparoscopic Liver Surgery  
  [2022] :star:[paper](https://link.springer.com/article/10.1007/s11548-022-02774-1) 
- Real-to-Sim Registration of Deformable Soft Tissue with Position-Based Dynamics for Surgical Robot Autonomy  
  [2021] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9561177) 
- Feature-Guided Nonrigid 3-D Point Set Registration Framework for Image-Guided Liver Surgery: From Isotropic Positional Noise to Anisotropic Positional Noise  
  [2020] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9123604) 
- An Optimal Control Problem for Elastic Registration and Force Estimation in Augmented Surgery  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_8) :sunny:[code](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_8)
- 【P2ILF】An objective comparison of methods for augmented reality in laparoscopic liver resection by preoperative-to-intraoperative image fusion from the MICCAI2022 challenge  
  [2025] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841524002962) :triangular_flag_on_post:[dataset](https://p2ilf.grand-challenge.org/)
- A methodology and clinical dataset with ground-truth to evaluate registration accuracy quantitatively in computer-assisted Laparoscopic Liver Resection  
  [2022] :star:[paper](https://www.tandfonline.com/doi/abs/10.1080/21681163.2021.1997642) :triangular_flag_on_post:[dataset](https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/)
- 【DEPOLL】An in vivo porcine dataset and evaluation methodology to measure soft-body laparoscopic liver registration accuracy with an extended algorithm that handles collisions  
  [2019] :star:[paper](https://link.springer.com/article/10.1007/s11548-019-02001-4) :triangular_flag_on_post:[dataset](https://www.ircad.fr/research/data-sets/respiratory-cycle-3d-ircadb-02-copy/)

## Unsupervised_Soft-tissue_Tracking
- Ada-Tracker: Soft Tissue Tracking via Inter-Frame and Adaptive-Template Matching  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10611030) :sunny:[code](https://github.com/wrld/Ada-Tracker)
- Surgical Tattoos in Infrared: A Dataset for Quantifying Tissue Tracking and Mapping  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10458702) :triangular_flag_on_post:[dataset](https://dx.doi.org/10.21227/w8g4-g548)
- Video-Based Soft Tissue Deformation Tracking for Laparoscopic Augmented Reality-Based Navigation in Kidney Surgery  
  [2024] :star:[paper](https://ieeexplore.ieee.org/document/10555430) 
- Evaluating Unsupervised Optical Flow for Keypoint Tracking in Laparoscopic Videos  
  [2024] :star:[paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12928/129281V/Evaluating-unsupervised-optical-flow-for-keypoint-tracking-in-laparoscopic-videos/10.1117/12.3006405.short#:~:text=Inspired%20by%20the%20%E2%80%9DWhat%20Matters%20in%20Unsupervised%20Optical,the%20context%20of%20tracking%20keypoints%20in%20laparoscopic%20videos.) 
- Tracker Learning Surgical Images By Self-Supervised Learning: An Enhanced Unsupervised Deep Tracking Approach  
  [2023] :star:[paper](https://ieeexplore.ieee.org/document/10255175)
- 【SurT】SurgT challenge: Benchmark of Soft-Tissue Trackers for Robotic Surgery  
  [2024] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841523002451) :triangular_flag_on_post:[dataset](https://surgt.grand-challenge.org/)
- 【OBRDataset】Online tracking and retargeting with applications to optical biopsy in gastrointestinal endoscopic examinations  
  [2016] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841515001449) :triangular_flag_on_post:[dataset](https://hamlyn.doc.ic.ac.uk/vision/)

## Surgical_Phase_Recognition
- C-ECT: Online Surgical Phase Recognition with Cross-Enhancement Causal Transformer  
  [2023] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10230841) 
- SKiT: a Fast Key Information Video Transformer for Online Surgical Phase Recognition  
  [2023] :star:[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SKiT_a_Fast_Key_Information_Video_Transformer_for_Online_Surgical_ICCV_2023_paper.html) :sunny:[code](https://github.com/MRUIL/SKiT)  
- 【Cholec80】EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos  
  [2016] :star:[paper](https://ieeexplore.ieee.org/abstract/document/7519080) :triangular_flag_on_post:[dataset](https://github.com/CAMMA-public/TF-Cholec80)
- SV-RCNet: Workflow Recognition From Surgical Videos Using Recurrent Convolutional Network  
  [2017] :star:[paper](https://ieeexplore.ieee.org/abstract/document/8240734) :sunny:[code](https://github.com/YuemingJin/SV-RCNet) :triangular_flag_on_post:[dataset](http://camma.u-strasbg.fr/datasets/)
- Temporal Memory Relation Network for Workflow Recognition From Surgical Video  
  [2021] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9389566) :sunny:[code](https://github.com/YuemingJin/TMRNet)
- Exploring Segment-Level Semantics for Online Phase Recognition From Surgical Videos  
  [2022] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9795918) :sunny:[code](https://github.com/XMed-Lab/SAHC)
- Cascade Multi-Level Transformer Network for Surgical Workflow Analysis  
  [2023] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10098668) 
- Federated Cycling (FedCy): Semi-Supervised Federated Learning of Surgical Phases  
  [2022] :star:[paper](https://ieeexplore.ieee.org/abstract/document/9950359) 
- LAST: LAtent Space-Constrained Transformers for Automatic Surgical Phase Recognition and Tool Presence Detection  
  [2023] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10136221) 
- Less Is More: Surgical Phase Recognition From Timestamp Supervision  
  [2023] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10043791) :sunny:[code](https://github.com/xmed-lab/TimeStamp-Surgical)
- Semi-supervised learning with progressive unlabeled data excavation for label-efficient surgical workflow recognition  
  [2021] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841521002048)
- LoViT: Long Video Transformer for surgical phase recognition  
  [2025] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841524002913) :sunny:[code](https://github.com/MRUIL/LoViT)
- Hard Frame Detection and Online Mapping for Surgical Phase Recognition  
  [2019] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_50) :sunny:[code](https://github.com/ChinaYi/miccai19)
- OperA: Attention-Regularized Transformers for Surgical Phase Recognition  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_58) 
- Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer  
  [2021] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_57) :sunny:[code](https://github.com/xjgaocs/Trans-SVNet) :triangular_flag_on_post:[dataset](http://camma.u-strasbg.fr/datasets/)
- HecVL: Hierarchical Video-Language Pretraining for Zero-Shot Surgical Phase Recognition  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_29) 
- Label-Guided Teacher for Surgical Phase Recognition via Knowledge Distillation  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_33) 
- MuST: Multi-scale Transformers for Surgical Phase Recognition  
  [2024] :star:[paper](https://repositorio.uniandes.edu.co/entities/publication/576ed23f-ac9a-4acd-9db0-7764ee326c4c) 
- Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition  
  [2024] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_57) :sunny:[code](https://github.com/isyangshu/Surgformer)
- Retrieval of Surgical Phase Transitions Using Reinforcement Learning  
  [2022] :star:[paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_47) 
- 【HeiChole】Comparative validation of machine learning algorithms for surgical workflow and skill analysis with the HeiChole benchmark  
  [2023] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000312) :triangular_flag_on_post:[dataset](https://www.synapse.org/Synapse:syn18824884/wiki/617550)
- On the pitfalls of Batch Normalization for end-to-end video learning: A study on surgical workflow analysis  
  [2024] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841524000513) :sunny:[code](https://gitlab.com/nct_tso_public/pitfalls_bn)
- Frequency-Based Temporal Analysis Network for Accurate Phase Recognition from Surgical Videos  
  [2024] :star:[paper](https://ieeexplore.ieee.org/abstract/document/10635806) 
- Intelligent surgical workflow recognition for endoscopic submucosal dissection with real-time animal study  
  [2023] :star:[paper](https://www.nature.com/articles/s41467-023-42451-8) :sunny:[code](https://github.com/med-air/AI-Endo)
- Dissecting Self-Supervised Learning Methods for Surgical Computer Vision  
  [2023] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841523001044) :sunny:[code](https://github.com/CAMMA-public/SelfSupSurg)
- Multi-task recurrent convolutional network with correlation loss for surgical video analysis  
  [2020] :star:[paper](https://www.sciencedirect.com/science/article/pii/S1361841519301124) :sunny:[code](https://github.com/YuemingJin/MTRCNet-CL)
- 【M2cai16】Tool Detection and Operative Skill Assessment in Surgical Videos Using Region-Based Convolutional Neural Networks  
  [2018] :star:[paper](https://arxiv.org/abs/1802.08774) :triangular_flag_on_post:[dataset](https://ai.stanford.edu/~syyeung/tooldetection.html)
- 【MISAW】MIcro-surgical anastomose workflow recognition challenge report  
  [2021] :star:[paper](https://www.sciencedirect.com/science/article/pii/S0169260721005265) :triangular_flag_on_post:[dataset](http://www.synapse.org/MISAW)








