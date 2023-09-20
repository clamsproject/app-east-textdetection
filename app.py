import argparse
import logging
from typing import Union, Sequence

import cv2
import itertools
import numpy as np
from clams import ClamsApp, Restifier
from mmif import Mmif, DocumentTypes, View, AnnotationTypes, Document
from mmif.utils import video_document_helper as vdh

from east_utils import image_to_east_boxes


class EastTextDetection(ClamsApp):
    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # see metadata.py
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        for videodocument in mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            # one view per video document
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            config = self.get_configuration(**parameters)
            new_view.new_contain(
                AnnotationTypes.BoundingBox,
                document=videodocument.id,
                timeUnit=config["timeUnit"],
            )
            self.logger.debug(f"Running on video {videodocument.location_path()}")
            mmif = self.run_on_video(mmif, videodocument, new_view, **config)
        if mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            # one view for all image documents
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            new_view.new_contain(AnnotationTypes.BoundingBox)
            self.logger.debug(f"Running on all images")
            mmif = self.run_on_images(mmif, new_view)

        if config["mergeBoxes"]:
            merge_view = mmif.new_view()
            self.sign_view(merge_view, parameters)
            merge_view.new_contains(AnnotationTypes.BoundingBox)
            mmif = self.merge_boxes(mmif)
        return mmif

    def get_boundary_box_coords(self, mmif: Mmif):
        """finds the four corner bounding boxes for each timepoint"""
        alignment_views = mmif.get_view_contains(AnnotationTypes.Alignment)
        bbox_views = mmif.get_view_contains(AnnotationTypes.BoundingBox)[0]
        timepoint_views = mmif.get_view_contains(AnnotationTypes.TimePoint)
        point_dict = defaultdict(list)

        annotation_id_to_annotation = {
            annotation.id: annotation for annotation in alignment_views.annotations
        }
        annotation_id_to_annotation.update(
            {annotation.id: annotation for annotation in bbox_views.annotations}
        )
        annotation_id_to_annotation.update(
            {annotation.id: annotation for annotation in timepoint_views.annotations}
        )

        for annotation in alignment_views.annotations:
            if annotation.at_type == AnnotationTypes.Alignment:
                timepoint_id = annotation.properties["source"]
                box_id = annotation.properties["target"]
                timepoint_anno = annotation_id_to_annotation[timepoint_id]
                box_anno = annotation_id_to_annotation[box_id]
                points_dict[timepoint_anno].append(box_anno)
        corners = defaultdict(list)
        for timepoint, bboxes in points_dict.items():
            # Since we know bbox is [TL, TR, BL, BR]
            curr_TL = [0, 0]
            curr_TR = [0, 0]
            curr_BL = [0, 0]
            curr_BR = [0, 0]
            for TL, TR, BL, BR in bboxes:
                if TL[0] <= curr_TL[0] and TL[1] <= curr_TL[1]:
                    curr_TL = TL
                if TR[0] >= curr_TR[0] and TR[1] <= curr_TR[1]:
                    curr_TR = TR
                if BL[0] <= curr_BL[0] and BL[1] >= curr_BL[1]:
                    curr_BL = BL
                if BR[0] >= curr_BR[0] and BR[1] >= curr_BR[1]:
                    curr_BR = BR
            corners[timepoint] = [TL, TR, BL, BR]
        return corners

    def merge_boxes(mmif: Mmif, view: View) -> Mmif:
        for time_point, box_coords in box_dict.items():
            bb_annotation = view.new_annotation(AnnotationTypes.BoundingBox)
            tp_annotation = view.new_annotation(AnnotationTypes.TimePoint)
            tp_annotation.add_property("timeUnit", config["timeUnit"])
            tp_annotation.add_property("timePoint", tp)

            bb_annotation.add_property("boxType", "text")

            bb_annotation.add_property("coordinates", box_coords)

            alignment_annotation = view.new_annotation(AnnotationTypes.Alignment)
            alignment_annotation.add_property("source", tp_annotation.id)
            alignment_annotation.add_property("target", bb_annotation.id)

        return mmif

    def run_on_images(self, mmif: Mmif, new_view: View) -> Mmif:
        for imgdocument in mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            image = cv2.imread(imgdocument.location)
            box_list = image_to_east_boxes(image)
            for idx, box in enumerate(box_list):
                annotation = new_view.new_annotation(
                    f"td{idx}", AnnotationTypes.BoundingBox
                )
                annotation.add_property("document", imgdocument.id)
                annotation.add_property("boxType", "text")
                x0, y0, x1, y1 = box
                annotation.add_property(
                    "coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
                )
            return mmif

    def run_on_video(
        self, mmif: Mmif, videodocument: Document, new_view: View, **config
    ) -> Mmif:
        cap = vdh.capture(videodocument)
        views_with_tframe = [
            v
            for v in mmif.get_views_for_document(videodocument.id)
            if v.metadata.contains[AnnotationTypes.TimeFrame]
        ]
        if views_with_tframe:
            frame_type = set(config["frameType"])
            frame_type.discard(
                ""
            )  # after this, if this set is empty, the next step will use "all" types
            # now for each of frame of interest, we will sample 2 (hard-coded) frames, evenly distributed
            target_frames = set()
            target_frames.update(
                *[
                    np.linspace(*vdh.convert_timeframe(mmif, a, "frame"), 2, dtype=int)
                    for v in views_with_tframe
                    for a in v.get_annotations(AnnotationTypes.TimeFrame)
                    if not frame_type or a.get_property("frameType") in frame_type
                ]
            )
            target_frames = list(map(int, target_frames))
            self.logger.debug(
                f"Processing frames {target_frames} from TimeFrame annotations of {frame_type} types"
            )
        else:
            target_frames = vdh.sample_frames(
                sample_ratio=config["sampleRatio"],
                start_frame=0,
                end_frame=min(
                    int(config["stopAt"]), videodocument.get_property("frameCount")
                ),
            )
        target_frames.sort()
        self.logger.debug(f"Running on frames {target_frames}")
        for fn, fi in zip(
            target_frames, vdh.extract_frames_as_images(videodocument, target_frames)
        ):
            self.logger.debug(f"Processing frame {fn}")
            result_list = image_to_east_boxes(fi)
            for box in result_list:
                bb_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                tp = vdh.convert(
                    time=fn,
                    in_unit="frame",
                    out_unit=config["timeUnit"],
                    fps=videodocument.get_property("fps"),
                )
                self.logger.debug(f"Adding a timepoint at frame: {fn} >> {tp}")

                tp_annotation = new_view.new_annotation(AnnotationTypes.TimePoint)
                tp_annotation.add_property("timeUnit", config["timeUnit"])
                tp_annotation.add_property("timePoint", tp)

                # bb_annotation.add_property("timePoint", tp)
                bb_annotation.add_property("boxType", "text")
                x0, y0, x1, y1 = box
                bb_annotation.add_property(
                    "coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
                )

                alignment_annotation = new_view.new_annotation(
                    AnnotationTypes.Alignment
                )
                alignment_annotation.add_property("source", tp_annotation.id)
                alignment_annotation.add_property("target", bb_annotation.id)

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = EastTextDetection()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
