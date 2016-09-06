# Download and Setup

This tutorial will guide installation of TensorFlow on either Linux, OSX and Windows.

# Docker

In this tutorial we will use [docker](https://www.docker.com/) containers to handle dependancies and run our code.
Docker is a container which allows us to run our code in an encapsuled container.
The language of choice will be python2

## Installation of docker for CPU (all operating systems)

Instructions for installing docker can be found [here](https://docs.docker.com/engine/installation/#installation), the intructions contains guides for most operating systems.

After installion and veryfying the docker installation, use the following [image](https://hub.docker.com/r/alrojo/tf-sklearn-cpu/), such that

>docker run alrojo/tf-sklearn-cpu

puts you into a terminal, which is the docker container.
To exit docker type

>exit

As the docker system runs independant of your host system, we need to enable port forwarding (for jupyter notebook) and sharing of directories.

First, make sure that you have downloaded this repository, if not, you can either go to `github.com/alrojo/tensorflow_tutorial`, click `Clone or download`, download as zip and extract to your desired folder.
Alternatively you can run the command

>git clone https://github.com/alrojo/tensorflow_tutorial.git

from now, the folder you chose to place your repository will be denoted as `$PATH\_TO\_FOLDER`.
And the name of the repository will be denoted as tensorflow_tutorial.
Given these namings, run the following line in your shell

>docker run -p 8888:8888 -v $PATH\_TO\_FOLDER/TensorFlowTutorial:/mnt/TensorFlowTutorial -it alrojo/tf-sklearn-cpu

where `-it` is required for an interactive experience with the docker bash environment, `-p` is for port forwarding	and `-v` is for mounting your given folder to the docker container.

This should leave you in the root directory of your docker container with port forwarded and shared directory, which you can access through the `/mnt` folder, run

>./run\_jupyter.sh

now go into you browser and navigate to `localhost:8888`, do note that you cannot have any other notebooks running simultaneously.

Navigate to the TensorFlowTutorial folder from you notebook, now you can start the exercises!

## Installation of nvidia-docker for GPU

To run neural nets on GPU accelerated hardware we use a slight modification of docker called [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (reason is that docker does not yet support the use-case of the specialised hardware and drivers we need).

Not yet supported %(need sudo access to a GPU server to test this)
