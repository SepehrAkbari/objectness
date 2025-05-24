# Faster-RCNN

This part of the project utilizes a pre-trained Faster R-CNN model to detect and propose regions of interest (potential objects) within paintings. The goal is to extract these regions as crops, which can then be used for further analysis, such as painter style identification. This is done in two primary steps:

1. **Region Proposal Network (RPN):** 
    * A deep convolutional network first processes the input image to extract rich feature maps.
    * The RPN then slides small networks over these feature maps to identify and propose candidate bounding boxes that are likely to contain objects. These proposals are class-agnostic at this stage, meaning the RPN suggests regions that look like "objects" in general, without knowing their specific class yet. It also provides an "objectness" score for each proposal.

2. **Detection Network (using a Faster-RCNN):**
    * The candidate regions proposed by the RPN are then passed to the second stage. For each region, features are pooled from the shared convolutional feature maps.
    * It processes each input painting, collecting the detected bounding boxes that have a confidence score above a defined threshold. The specific class labels are not the primary focus here, only the presence of an "object."
    * A Non-Maximum Suppression (NMS) algorithm is then applied to reduce redundant, highly overlapping proposals.
    * Finally, a limited number of the highest-confidence, distinct proposals are selected and cropped from the original painting.

This approach allows us to obtain semantically relevant crops based on what a powerful deep learning model considers an object.

## Usage

Clone the repository and navigate to the Deep Learning method's directory:

```bash
git clone https://github.com/SepehrAkbari/objectness.git
cd objectness/method_DeepLearning
```

Then, simply run the script:

```bash
python rp_rcnn.py
```

The results will be saved in the `output` directory.