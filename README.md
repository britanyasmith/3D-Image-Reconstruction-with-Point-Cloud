# 3D-Image-Reconstruction-with-Point-Cloud

The reconstruction of 3D objects from 2D images is
a growing topic in machine learning, and the problem proves to
be challenging as we are faced with computations that grow
in cubic form as we attempt to process larger inputs. 3D-
ReConstnet is a point-cloud-based 3D reconstruction algorithm
that aims to eliminate the concerns with voxel-based occupancy
grid computational complexity. In this project, we want to explore
the architecture of 3D-ReConstnet and recreate the network
by hand. After recreating the network, we want to explore
improvements and optimizations, so that we can improve the
resulting point-cloud generation.

## The outline of the steps performed during this project
implementation is as followed:
1) Obtain 2D images from the ShapeNet Dataset [3] and
extract its necessary features to a feature vector using the
ResNet-50 Network. Different combinations of input to
the network will be experimented with to obtain different
outputs.
2) Obtain the mean and standard deviation of the feature
vector and obtain the Gaussian probabilistic vector to
encode the results.
3) Perform decoding of the Probabilistic Vector by imple-
menting a multi-layer perception.
4) Experiment with different loss functions and training
parameters to improve 3D results of output image. 

<p align="center">
  <img src="/pointCloud.png" />
</p>

