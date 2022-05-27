# Footwork Recognition
Repository for development of footwork action recognition model for fencing video data. 

Initially videos are downloaded from youtube and cut up into clips that contain scored touches using code found in [data_aquisition](https://github.com/Carpfire/footwork_recognition/tree/main/src/data_aquisition)

We then train a detection model to detect fencers, referees and the crows using detectron 2. An out of the box pose estimation model is run on the detected fencers and the pose data is saved. 

![Fencer Detection](https://github.com/Carpfire/footwork_recognition/blob/main/Bardenet_Cannone.gif Example Fencer & Referee Detection)

![Initial Pose Estimation](https://github.com/Carpfire/footwork_recognition/blob/main/bardenet_pose.gif Initial Pose Estimation)

Poses are noticably noisy due to the lack of domain specific videos, for fencing in this case, in the training corpus of the original model.
These poses are refined by training a student teacher network using high confidence poses as a supervision signal, as done in [Video Pose Distillation](https://github.com/jhong93/vpd). 

![VPD Pose Esitmation](https://github.com/Carpfire/footwork_recognition/blob/main/bardenet_pose_vpd.gif VPD Pose Estimation)

These refined poses are then used as features in a GRU Neural Network and trained to predict on footwork classes in [baseline.ipynb](https://github.com/Carpfire/footwork_recognition/blob/main/src/baseline.ipynb). 

[Example annotations and embeddings ](https://drive.google.com/drive/folders/1qpjcw3qh63K0SM9M9iTy7TSdffHYdNkd?usp=sharing)

[Example Colab Notebook](https://colab.research.google.com/drive/1Bo0T0Wm9TStcp2kologhsT9a3sYZ4M2z#scrollTo=jjmWfDXTXcXK)
