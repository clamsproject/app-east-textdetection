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
        description="OpenCV-based text localization app that used EAST text detection model. Please visit the source code repository for full documentation.",
        app_license="Apache 2.0",
        identifier="east-textdetection",
        url="https://github.com/clamsproject/app-east-textdetection",
    )
    metadata.add_input_oneof(DocumentTypes.VideoDocument, DocumentTypes.ImageDocument)
    metadata.add_input(AnnotationTypes.TimeFrame, required=False)
    metadata.add_output(AnnotationTypes.BoundingBox, label="text")

    metadata.add_parameter(
        name="timeUnit", 
        type="string", 
        choices=["frames", "seconds", "milliseconds"],
        default="frames",
        description="Unit for time points in the output. Only works with VideoDocument input.",
    )
    metadata.add_parameter(
        name="frameType",
        type="string",
        choices=["", "slate", "chyron", "rolling-credit"],
        default="",
        multivalued=True,
        description="Segments of video to run on. Only works with VideoDocument input and TimeFrame input. Empty value means run on the every frame types.",
    )
    metadata.add_parameter(
        name="sampleRate",
        type="integer",
        default=30,
        description="Frequency to sample frames. Only works with VideoDocument input, and without TimeFrame input. (when `TimeFrame` annotation is found, this parameter is ignored.)",
    )
    metadata.add_parameter(
        name="stopAt",
        type="integer",
        default=2 * 60 * 60 * 30,  # ~2 hours of video at 30fps 1 * 60 * 60 * 30
        description="Frame number to stop running. Only works with VideoDocument input. The default is roughly 2 hours of video at 30fps.",
    )
    metadata.add_parameter(
        name="mergeBoxes",
        type="boolean",
        default=False,
        description="if True, creates a single merged bounding box from all detected boxes."
    )
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
