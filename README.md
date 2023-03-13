# Jain Software SLA Task 1 

### Measuring Pupil Distance (PD) from an image using card on forehead

By : [Tanishq Selot](https://github.com/tanishq150802)

## Requirements
* Python 3.10
* Numpy
* dlib
* OpenCV
* imutils

All the code is contained within **pupil_detect.ipynb**. Image examples can be seen in the folders (each one has its own README.md).

### Steps

* The card on forehead is detected using HSV and Adaptive thresholding.
* A bounding box is drawn around the card and its width is calculated.
* The left and right pupil coordinates are detected using dlib library.
* The PD is calculated in terms of pixels using Euclidean distance.
* The width of a credit card is 8.56 cm.
* This data along with scaling formula is used to calculate the PD in cm.
* A variable PARALLEL_PLANE_SCALING_FACTOR is multiplied to take into consideration the depth and camera calibration aspects.
* PARALLEL_PLANE_SCALING_FACTOR is 1.372 in my case.

### Eaxample

Normal Image             |  Pupil Line with PD |  Card Detected
:-------------------------:|:-------------------------: |:-------------------------:
![img6](https://user-images.githubusercontent.com/81608921/224797617-1426fcd8-0de7-49f4-a5e1-7467015395bb.jpeg) |  ![result](https://user-images.githubusercontent.com/81608921/224797956-d5023fc1-cd18-4abb-b061-3af0bbd2d97a.jpg) |  ![card_detected_1](https://user-images.githubusercontent.com/81608921/224798264-fbeaefe7-bdee-49fc-81b2-3986b98ae37d.jpg)

### References
* [Pupil Detection](https://github.com/weblineindia/AIML-Pupil-Detection)
* [Scaling formula](https://www.youtube.com/watch?v=ghU6T4h-C74)
