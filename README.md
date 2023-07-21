## User instruction

This app is a text localization app based on OpenCV and EAST text detection model. 

General user instruction for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### Usage

This is an OpenCV-based text localization app that used EAST text detection model. 
Text *localization* is a technique to detect the location of "scene text" in an image or video. "

The app implementation is based on [EAST algorithm](https://arxiv.org/abs/1704.03155), and the frozen EAST model is downloaded from this tutorial: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/ .

This app can process both images and videos and when analyzing videos, it can run in two modes: segment-based mode and resampling mode (these modes are irrelevant when analyzing images).

#### Segment-based mode
When the app finds existing `TimeFrame` annotation (Please visit https://mmif.clams.ai/ for details about CLAMS annotation types) in the input MMIF, it takes them as segmentation and runs in segment-based mode. In segment-based mode the app extracts 2 frames (the first and the last) from one segment, then perform text localization on these two frames.

##### Relevant parameter
* `frameType`: the type of segment to use (default is empty value (`""`) and it means using all types of segments)

#### resampling mode
When there's no `TimeFrame` annotation to use, the app automatically runs in resampling mode. In resampling mode, the app extracts frames from the input video at a fixed interval and perform text localization on these frames.

##### Relevant parameter
* `sampleRatio`: the sampling interval in frames numbers (default is 30, which means extracting one frame every 30 frames)

#### More parameters

For the full list of parameters, please refer to the app metadata from [CLAMS App Directory](https://apps.clams.ai/clamsapp/) or [`metadata.py`](metadata.py) file in this repository.

### System requirments
* [OpenCV 4](https://docs.opencv.org/4.x/index.html)
 