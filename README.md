# Denoising Diffusion Probabilistic Models 

## Description

PyTorch implementation of the DDPM paper by [Ho et al. \[2020\]](https://arxiv.org/abs/2006.11239).

Generative model that works by iteratively removing normally distributed noise from a standard normal distribution until an image is formed, referred to as the reverse process.
The forward process used to generate training data is done by starting with an image from a dataset and adding noise to it.

Algorithm 1 and 2 from the paper are implemented in the DDPM class.

<div align="center">
    <img src="Images/DDPM_algorithms.png" height="200">
</div>

The model used to learn the added noise is a fully convolutional U-Net based on a series of downsampling and upsampling layers with residual blocks at each resolution.
Multi-headed self attention is also added at certain steps, and the time-dependency is added to each residual block using a sinusoidal positional embedding.

## Training

Example batch from the MNIST training dataset.

<div align="center">
    <img src="Images/MNIST.png" height="500">
</div>

Before any training, sampling from the model leads to images mostly looking like random noise. After training the model on the MNIST training dataset for 10 epochs, samples start to resemble the images from the training.

<div align="center">
    <img src="Images/DDPM_random_samples.png" height="300">
    <img src="Images/DDPM_generated_samples.png" height="300">
</div>


## Conditional DDPM

Classifier-Free Diffusion Guidance implementation based on the paper by [Ho & Salimans \[2022\]](https://arxiv.org/abs/2207.12598).

Adds class conditioning to the original DDPM version. Makes it possible to choose a class and generate images from that given class.
This is done by adding the class context to the residual blocks of the U-Net for a guided prediction step.
The final noise predictions at each step is computed as a weighted sum between a guided and unguided prediction for sampling images.

Example images from each class are shown in the table below after training.

<div align="center">
    <table>
        <tr>
            <td><img src="Images/CDDPM_0.png", height=100></td>
            <td><img src="Images/CDDPM_1.png", height=100></td>
            <td><img src="Images/CDDPM_2.png", height=100></td>
            <td><img src="Images/CDDPM_3.png", height=100></td>
            <td><img src="Images/CDDPM_4.png", height=100></td>
        </tr>
        <tr>
            <td><img src="Images/CDDPM_5.png", height=100></td>
            <td><img src="Images/CDDPM_6.png", height=100></td>
            <td><img src="Images/CDDPM_7.png", height=100></td>
            <td><img src="Images/CDDPM_8.png", height=100></td>
            <td><img src="Images/CDDPM_9.png", height=100></td>
        </tr>
    </table>
</div>
