# FASTMAD
This work accelerate the calculating speed of 'Most Apparent Distortion, MAD' published in "Larson E C, Chandler D M. Most apparent distortion: full-reference image quality assessment and the role of strategy[J]. Journal of electronic imaging, 2010, 19(1): 011006."

We employ the convolutional operation to optimize the time-consuming procedure of block-wise statistical information extraction (e.g., mean, std, skr...)

Experiment results indicate the FAST-MAD takes only 0.3s to annotate a HD image, whilst original version in matlab needs around 6s (Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz 3.19 GHz & 32G RAM & 1080Ti GPU)

Python3 and Tensorflow 2.1 is needed to run the script.

To begin with our could, modify the 'kadis700k_names.csv' so that your refrence images are listed in its 'ref_im' columns and your distorted images are listed in its 'dist_im' columns.
