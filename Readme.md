
# Neural Style Transfer (NST) Project

  

## Project Overview

This project implements Neural Style Transfer (NST) using deep learning techniques. NST is a method of blending two images, a content image and a style image, to create a new image that retains the content of the first image but adopts the artistic style of the second image. This project utilizes convolutional neural networks (CNNs) to achieve this effect.

  

## Installation Instructions

To set up the environment and run the project, follow these steps:

  

For *Python3* users just use *pip3* and *python3* respectively

  

1. **Clone the repository**:

```bash

git clone https://github.com/yourusername/nst-project.git

cd nst-project

  ```

2. **Create a virtual environment (optional but recommended)**:

```bash

python -m venv venv

source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

3. **Install the required dependencies**:

```bash

pip install tensorflow Flask numpy pillow scipy tensorflow-hub matplotlib

```

4. **Run the following code on terminal**

> You can also run the code in any IDE, but ensure to set the interpreter to the virtual environment created above. If you already have all the required packages installed, you can use the default environment. 
```bash
python App.py
```



5. **Open the `index.html` file** in the `nst-project` directory to access the web interface where you can upload images to perform style transfer. Resulting images are stored in the `Generated_Image` folder within the project directory.



