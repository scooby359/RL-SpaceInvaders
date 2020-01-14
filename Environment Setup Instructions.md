# Environment Setup

## Dependent Software

The environment needs the following software to be installed:

### [Pycharm Community Edition](https://www.jetbrains.com/pycharm/)

Used as it creates a container for each project, allowing you to select the Python and dependency versions for your project, without conflicting with any global settings.

### [Atari Roms](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)

Used for the simulation. Only require one game, but this provides all.

### [Python 3.6 for Windows 64-bit](https://www.python.org/ftp/python/3.6.8/python-3.6.8.exe)

No need to add this to path

### [Git for Windows](https://gitforwindows.org/)

For version control support

### [CUDA Toolkit 9.0]( https://developer.nvidia.com/cuda-90-download-archive)

Used by Tensorflow for GPU support

### [cuDNN 7 *](https://developer.nvidia.com/rdp/cudnn-download)

The file `cudnn64_7.dll` is needed by Tensorflow for GPU support. The download should be extracted and placed somewhere on drive. The path to the DLL will must be manually added to the System Environment PATH variable.

\* Registration to NVIDIA required

## Process

It is recommended to use a machine with a compatabile NVIDIA graphics card that can be used by TensorFlow. The UCLan Games Lab has PCs with suitable NVIDIA GeForce GTX 1070 cards.

Not using a graphics card is significantly slower and not recommended.

1. Open Pycharm and checkout code from Github <https://github.com/scooby359/RL-SpaceInvaders.git>

2. When prompted by PyCharm, set the environment to Python 3.6, or manually go to the project options and set.

3. From the PyCharm terminal, verify Pip is available with `pip -V`

4. Install packages for skimage (scikit-image), matplotlib by using the prompts in PyCharm.

5. Install TensorFlow 1.10 with `pip install tensorflow-gpu==1.10`. If not using a machine with a suitable graphics card, install `tensorflow==1.10` instead.

6. Install Numpy with `pip install numpy==1.17.4`
Tensorflow will require version <= 1.14.5, but this isn't compatible with the rest of the project so needs to be manually changed.

7. Install OpenAI Retro with `pip install gym-retro`

8. Import the roms by extracting the roms RAR file into `.\roms\`. Then from the terminal, run `python -m retro.import ./roms`

9. The project can then be run from PyCharm by right clicking on the Python file and selecting Run.
