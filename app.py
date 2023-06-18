import argparse
from typing import Union

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier
from mmif import Mmif, DocumentTypes, View, AnnotationTypes

from east_utils import *
from east_utils import image_to_east_boxes


class EastTextDetection(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        for videodocument in mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            # one view per video document
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            config = self.get_configuration(**parameters)
            new_view.new_contain(AnnotationTypes.BoundingBox, timeUnit=config["timeUnit"])
            mmif = self.run_on_video(mmif, videodocument, new_view, **config)
        if mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            # one view for all image documents
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            new_view.new_contain(AnnotationTypes.BoundingBox)
            mmif = self.run_on_images(mmif, new_view)
        return mmif

    def run_on_images(self, mmif: Mmif, new_view: View) -> Mmif:
        for imgdocument in mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            image = cv2.imread(imgdocument.location)
            box_list = image_to_east_boxes(image)
            for idx, box in enumerate(box_list):
                annotation = new_view.new_annotation(f"td{idx}", AnnotationTypes.BoundingBox)
                annotation.add_property("document", imgdocument.id)
                annotation.add_property("boxType", "text")
                x0, y0, x1, y1 = box
                annotation.add_property(
                    "coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
                )
            return mmif

    def run_on_video(self, mmif: Mmif, new_view: View, **kwargs) -> Mmif:
        cap = cv2.VideoCapture(mmif.get_document_location(DocumentTypes.VideoDocument))
        frame_type = kwargs["frameType"]
        views_with_tframe = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
        if views_with_tframe:
            target_frames = self.get_target_frame_numbers(views_with_tframe, frame_type, 2)
        else:
            target_frames = range(0, min(int(kwargs['stopAt']), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), kwargs['sampleRatio'])
        self.boxes_from_target_frames(target_frames, cap, new_view, kwargs["timeUnit"])
        return mmif

    @staticmethod
    def boxes_from_target_frames(target_frames: List[int], cap: cv2.VideoCapture, new_view:View, output_unit: str):
        for frame_number in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _, f = cap.read()
            result_list = image_to_east_boxes(f)
            for box in result_list:
                bb_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                if output_unit == "frames":
                    timepoint = frame_number
                elif output_unit == "seconds":
                    timepoint = frame_number / cap.get(cv2.CAP_PROP_FPS)
                elif output_unit == "milliseconds":
                    timepoint = frame_number / cap.get(cv2.CAP_PROP_FPS) * 1000
                else:
                    raise ValueError(f"Invalid output time unit: {output_unit}")
                bb_annotation.add_property("timePoint", timepoint)
                bb_annotation.add_property("boxType", "text")
                x0, y0, x1, y1 = box
                bb_annotation.add_property("coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]])

    @staticmethod
    def get_target_frame_numbers(views_with_tframe, frame_types, frames_per_segment=2):
        def convert_msec(time_msec):
            import math
            return math.floor(time_msec * 29.97)  # todo 6/1/21 kelleylynch assuming frame rate

        frame_number_ranges = []
        for tf_view in views_with_tframe:
            for tf_annotation in tf_view.get_annotations(AnnotationTypes.TimeFrame):
                if not frame_types or tf_annotation.properties.get("frameType") in frame_types:
                    frame_number_ranges.append(
                        (tf_annotation.properties["start"], tf_annotation.properties["end"])
                        if tf_view.metadata.get_parameter("timeUnit") in ["frames", "frame"]
                        else (convert_msec(tf_annotation.properties["start"]), convert_msec(tf_annotation.properties["end"]))
                    )
        target_frames = list(set([int(f) for start, end in frame_number_ranges
                                  for f in np.linspace(start, end, frames_per_segment, dtype=int)]))

        return target_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = EastTextDetection()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
