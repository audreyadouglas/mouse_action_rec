# Fast Localization and Slow Classification of Mouse Behavior
Prediction using Sparse Video with Trajectory Estimation
Created by Audrey Douglas for University of Michigan CSE 598

# Overview
You can find more about the project at https://audreyadouglas.github.io/mouse_action_rec/

Manual labeling of animal behavior, such as in mice studies, is timeconsuming, inconsistent, and limits large-scale analysis. This is a task that would be revolutionized by machine labeling. Although recent advances in video understanding with transformer-based vision-language models (VLMs) enable richer temporal embeddings, they remain computationally infeasible for continuous video and lose information over long videos. To address this, we propose a two-stage temporal action localization pipeline for mouse behavior recognition. In the first stage, features, either from pose estimation or frame embeddings, are used to identify candidate action segments. In the second stage, these segments are processed with the Qwen2.5-VL video VLM to obtain action segment embeddings, which are then classified using either a neural network (for actions with many labels) or clustering (for actions with few labels). We evaluate our method with pose embedding features on both a supervised and few shot learning mouse behavior dataset. By combining low-cost action localization with high-accuracy VLMbased classification, our approach aims to reduce computation while maintaining performance, offering a scalable method for predicting mouse behavior in long or continuous video recordings.

This project does action recognition through a 2 phase anchor method.

**Stage 1**  
![arch_stage1](https://github.com/audreyadouglas/mouse_action_rec/blob/main/DOC/arch_stage1.png)

**Stage 2**  
![arch_stage2](https://github.com/audreyadouglas/mouse_action_rec/blob/main/DOC/arch_stage2.png)
The VLM being used for this project is Qwen2-VL-2B-Instruct, but other models can be easily substituted.

# Setup
Qwen-2.5 requires being run on a GPU, so make sure you have the proper resources. On the Great Lakes compute cluster use the packages mamba/py3.10 and ffmpeg.

For all other parts of the pipeline use pip install requirements.txt.

You can donwload the necessary data from <https://data.caltech.edu/records/s0vdx-0k302>.

# Repo Overview
To train and run the model through the pipeline data must be passed through the following files, in order.
**pred_task1.py** - Trains stage 1 of the network to predict if an action happens.  
**predict_ifaction.ipynb** - Use model trained from pred_task1.py to create video clips.  
**video_embedding_qwen2.ipynb** - Feed in video clips from predict_ifaction.ipynb to create embeddings.  
**action_pred_network.ipynb** - Use embeddings from video_embedding_qwen2.ipynb to train action prediction network.  
**action_pred_network.ipynb** - Using model from action_pred_network.ipynb or clustering predicts actions and evaluates model.

# Demo
To try a quick demo with a set video and pretrained models, use demo/demo.ipynb.
