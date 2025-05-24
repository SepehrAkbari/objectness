# Binarized Normed Gradients

This method is inspired by [BING: Binarized Normed Gradients for Objectness Estimation at 300fps](https://mmcheng.net/bing/) by Ming-Ming Cheng et al. OpenCV's saliency module is used to implement the BING method, using the same weights described in the paper for HSV, MAXBGR, and B&W color channels. The BING method is a fast and efficient object-ness detection algorithm that uses binarized normed gradients to identify potential object locations in images.

## Usage

Clone the repository ad navigate to the BING method's directory:

```bash
git clone https://github.com/SepehrAkbari/objectness.git
cd objectness/method_BING
```

Before running any of the code, make sure you have the saliency module from OpenCV installed. Then, run the following command to compile the code:

```bash
cd build
cmake ../src
make
```

After the build is complete, you can run the BING method on images using the following command:

```bash
./CropperBING
```

This will produce a folder containing crops of the identified objects in the images, along with a CSV file with their coordinates. The results will be saved in the `output` directory.

**Note**:

1. In `src/bing_cropper.cpp`, you can change the `images_path` to match your images directory.

2. This module returns a fixed, 15 crops per image. You can change this in the `src/bing_cropper.cpp` file by modifying the parameters on Line 98. The value used is 15, but you can set it to any number you want.

3. `Base`, `W`, and `NSS` were set according to the paper. You can change them in `src/bing_cropper.cpp`, Line 41-43.