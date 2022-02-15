# StyleGAN 2 in PyTorch


## Requirements

I have tested on:

- PyTorch 1.3.1
- CUDA 10.1/10.2

## Usage
Download FFHQ and create a txt document with all image paths in ffhq.txt
Download IR-SE50 Model from (https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)
Download pretrained StyleGAN2 model from (https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)

Then you can train model in distributed settings

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_1024.py --batch BATCH_SIZE 

tensorboard logs saved to runs_1024
