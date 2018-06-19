# Deep photo style transfer

preventing spatial distortion and constraining the transfer operation to happen only in color space

- Structure preservation
  Keep photorealistic images, different from paintings

- Semantic accuracy
  Match building with buildings and sky with sky
  CNNMRF method

Semantic labeling of the input and style image so the tranfer happens between smantically equivalent subregions.

Global style tranfer is effective on simple styles like global color shifts (sepia) and tone curves.

"Neural Style computes global statistics of the reference style image which tends to produce texture mismatches as shown in the correspondence"

"CNNMRF computes a nearest-neighbor search of the reference style image which tends to have many-to-one mappings as shown in the correspondence"

Augment NS algorith with 2 ideas:
1. photorealism regularization term in the
objective function during the optimization to prevent distortions

2. Semantic segmentation
improve the photorealism

"Characterizing the space of photorealistic images is
an unsolved problem"

Add term to equation to penalizes image distortions

Add segmentation masks to the input image as additional channels

Shih’s hallucination needs a full time-lapse video.
