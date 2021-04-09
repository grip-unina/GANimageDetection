## GAN Image Detection
This repository contains the best GAN generated image detector, namely <*ResNet50 NoDown*>, among those presented in the paper:

**Are GAN generated images easy to detect? A critical analysis of the state-of-the-art**  
Diego Gragnaniello, Davide Cozzolino, Francesco Marra, Giovanni Poggi and Luisa Verdoliva.
<br />In IEEE International Conference on Multimedia and Expo (ICME), 2021.

The very same architecture has been trained with images generated either by the Progressive Growing GAN or the StyleGAN2 architecture.
<br />Download the trained weights from [here](https://www.grip.unina.it/download/GANdetection) and put in the folder *weights*.

### Requirements
- python>=3.6
- numpy>=1.19.4
- pytorch>=1.6.0
- torchvision>=0.7.0
- pillow>=8.0.1

### Test on a folder

To test the network on an image folder and to collect the results in a CSV file, run the following command:

```
python main.py -m weights/gandetection_resnet50nodown_stylegan2.pth -i ./example_images -o out.csv
```

