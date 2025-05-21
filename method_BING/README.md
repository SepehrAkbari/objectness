# Binarized Normed Gradients

This method is inspired by [BING: Binarized Normed Gradients for Objectness Estimation at 300fps](https://mmcheng.net/bing/) by Ming-Ming Cheng et al. OpenCV's saliency module is used to implement the BING method, using the same weights described in the paper for HSV, MAXBGR, and B&W color channels. The BING method is a fast and efficient objectness detection algorithm that uses binarized normed gradients to identify potential object locations in images.

## Usage

Clone the repository ad navigate to the BING method's directory:

```bash
git clone https://github.com/SepehrAkbari/objectness.git
cd objectness/method_BING
```

Before running any of the code, make sure you have the saliency module from OpenCV installed. This module is not included in many of the OpenCV distributions, we recommend installing OpenCV with homebrew.

```bash
brew install opencv
```

Then, run the following command to compile the code:

```bash
cd build
cmake ../src
```
After that, run the following command to build the code:

```bash
make
```

After the build is complete, you can run the BING method on images using the following command:

```bash
./CropperBING
```

The results will be saved in the `output` directory.