# psfFold



A project focused on generating and experimenting with camera Point Spread Functions (PSF) for image realism, optical simulation, and synthetic data generation.



## Project Purpose

This repository contains tools and experiments for simulating realistic camera blur:

- Gaussian PSF

- Airy disk (diffraction)

- Motion blur

- Atmospheric scattering

- Sensor crosstalk



The goal is to create realistic synthetic images that match real camera behavior.



## Installation Instructions

Create a dedicated environment:



micromamba create -n psfEnv python=3.11

micromamba activate psfEnv



Install required libraries:



pip install scipy scikit-image numpy opencv-python pillow



## Usage Examples

Run PSF scripts:



python psf\_gaussian.py

python psf\_motion.py

python psf\_airy.py



## Dataset Description

This project works with any RGB or grayscale images.  

You can place your test images inside the folder and apply PSF kernels to simulate camera behavior.



## Future Plans

- Add physically accurate PSF based on real camera parameters

- Add atmospheric blur models

- Add batch processing for large datasets

- Add visualization tools for PSF kernels

- Add integration with Stable Diffusion for realistic synthetic data



