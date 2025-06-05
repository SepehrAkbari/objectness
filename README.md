# Objectness

This repository contains a module for detecting objects in images, and producing context-aware and meaningful crops. The initial problem trying to solve is to generate meaningful image crops for training a neural network for classification, when each class does not have enough images, and augmentation methods are not sufficient or possible.

In this case, our goal is to classify paintings from 100 different artists and multiple style classes. Each class has only about 30 paintings, and augmentation through rotation, color jittering, etc. is not possible due to the nature of the problem. This module is designed to help produce meaningful crops of the paintings, which can identify the most standout objects in a painting, producing a high quality training set for the neural network to learn contextual information about the style, and the artist themself.

## Approach

To design this module, we use a combination of two methods to detect objects in images. Both methods would then be orchestrated by a script to produce the final output. The first method utilizes a Faster-RCNN model with a Region Proposal Network (RPN) to detect objects through a Non-Maximum Suppression (NMS) algorithm; refer to the [Deep Learning Method](method_DeepLearning) directory for more details. The second method is aspired by the work of Ming-Ming Cheng et al. in their paper [BING: Binarized Normed Gradients for Objectness Estimation at 300fps](https://mmcheng.net/bing/), which introduces a fast and efficient method for object detection using Binarized Normed Gradients (BING). This method is implemented in the [BING Method](method_BING) directory using OpenCV's `saliency` module. For enhanced neural network training, we also include a small portion of low-saliency crops to provide a more diverse training set.

## Usage

To use the module, make sure you have a version of Python 3.8 or higher, Go-lang, and a C compiler installed, along with the OpenCV saliency module which is included in their full package.

First, clone the repository:

```bash
git clone https://github.com/SepehrAkbari/objectness.git
cd objectness
go mod init objectness
go mod tidy
```

Populate the [images](images) directory with your images, following the structure outlined. Then, install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First compile the BING source code:

```bash
cd main/bing_processor/build
cmake ../src
make
cd ../..
```

You can then build and run the orchestrator script to process the images:

```bash
go build orchestrator.go
./orchestrator
```

This will process the images in the `images` directory and save the output in the `main/output` directory. The output will include crops of the detected objects in the images, and just a few crops of objectless regions, along with a CSV file containing the coordinates of the detected objects, and other relevant information.

Because of hardware limitations, the orchestrator script is designed to be executed in a GPU-enabled environment, since this is usually done on an external machine, you can use the `main/cropper_helper.py` helper script to generate the crops based on the outputted CSV file by the orchestrator on your local machine, which saves significant time since there is no more need to transfer the images back and forth.

```bash
python cropper_helper.py
```

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](LICENSE).
