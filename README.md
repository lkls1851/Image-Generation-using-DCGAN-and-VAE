# Image-Generation-using-DCGAN-and-VAE
Winter in Data Science

This is the official Repository for my WIDS project on Deep Generative Models.
In this project, I have implemented the Deep Convolutional Generative ADversarial Network (DCGAN) on Celeba Dataset and Variational AutoEncoder (VAE) on the MNIST Dataset.
Further, in this project, I explored different Probabilistic Generative Models.

One such model is CycleGAN which I used for implementing Neural Style Transfer; converted images to 'Monet' style.
Further, I explored energy based methods, including implementation of a very basic UNet based Denoising Diffusion Probabilistic Model (DDPM). I have used DDPM on Gravitational Lensing Dataset to generate realistic simulations.

### Results for DDPM:
1. ‘ddpm_weights.pth’ : Contains the trained weights of the DDPM model.
2. ‘Generated’ : Contains grayscale images of generated images.
3. ‘Actual’ : Contains randomly sampled images from actual dataset.
4. ‘DDPM.ipynb’ : Notebook for implementing actual code
For calculating FID values for camparing the distribution in generated and actual images,
Run the following commands in the directory: </br>
pip3 install pytorch-fid </br>
python3 -m pytorch_fid Generated Actual --device cuda:0 </br>

The trained weights in the form of PyTorch weights uploaded: https://drive.google.com/file/d/1PuTiw5279Zh54HLYtJO8qVOfEvARdh5J/view?usp=sharing

