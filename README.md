# Fast Localization and Slow Classification of Mouse Behavior
Prediction using Sparse Video with Trajectory Estimation
Created by Audrey Douglas for University of Michigan CSE 598

# Overview
You can find more about the project at https://audreyadouglas.github.io/mouse_action_rec/

This project does action recognition through a 2 phase anchor method.

**Stage 1**  
![arch_stage1](https://github.com/audreyadouglas/mouse_action_rec/blob/main/website_resources/arch_stage1.png)

**Stage 2**  
![arch_stage2](https://github.com/audreyadouglas/mouse_action_rec/blob/main/website_resources/arch_stage2.png)
The VLM being used for this project is Qwen2-VL-2B-Instruct, but other models can be easily substituted.

# Setup
Qwen-2.5 requires being run on a GPU, so make sure you have the proper resources. On the Great Lakes compute cluster use the packages mamba/py3.10 and ffmpeg.

For all other parts of the pipeline use pip install requirements.txt

# Repo Overview
**pred_task1.py** - Trains stage 1 of the network to predict if an action happens.  
**predict_ifaction.ipynb** - Use model trained from pred_task1.py to create video clips.  
**video_embedding_qwen2.ipynb** - Feed in video clips from predict_ifaction.ipynb to create embeddings.  
**action_pred_network.ipynb** - Use embeddings from video_embedding_qwen2.ipynb to train action prediction network.  
**action_pred_network.ipynb** - Using model from action_pred_network.ipynb or clustering predicts actions and evaluates model.

# Demo
To try a quick demo with a set video and pretrained models, use demo.ipynb.
