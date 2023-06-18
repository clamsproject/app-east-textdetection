"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="EAST Text Detection",
        description="OpenCV-based text localization app that used EAST text detection model. "
                    "Text localization is a technique to detect the location of \"scene text\" in an image or video. "
                    "The frozen EAST model is downloaded from this tutorial: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/",
        app_license="Apache 2.0",
        identifier="east-textdetection",
        url="https://github.com/clamsproject/app-east-textdetection",
    )
    metadata.add_input_oneof(DocumentTypes.VideoDocument, DocumentTypes.ImageDocument)
    metadata.add_input(AnnotationTypes.TimeFrame, required=False)
    metadata.add_output(AnnotationTypes.BoundingBox, bboxtype="string")
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(AnnotationTypes.TimePoint)

    metadata.add_parameter(
        name="timeUnit", 
        type="string", 
        choices=["frames", "milliseconds"],
        default="frames",
        description="Unit for time points in the output. Only works with VideoDocument input.",
    )
    metadata.add_parameter(
        name="frameType",
        type="string",
        choices=["", "slate", "chyron"],
        default="",
        multivalued=True,
        description="Segments of video to run on. Only works with VideoDocument input and TimeFrame input. Empty value means run on the every frame types.",
    )
    metadata.add_parameter(
        name="sampleRatio",
        type="integer",
        default="30",
        description="Frequency to sample frames. Only works with VideoDocument input.",
    )
    metadata.add_parameter(
        name="stopAt",
        type="integer",
        default="540000",  # appr. 5 hours
        description="Frame number to stop running. Only works with VideoDocument input.",
    )
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(appmetadata().jsonify(pretty=True))
