import argparse
import logging
from typing import Union, Sequence

import copy
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
        """Internal Annotate Wrapper Method
        
        Generates a new set of annotations for `mmif` 
        via EAST Text Detection on Videos and Images. 

        ### params 
        + mmif => a mmif object
        + **parameters => runtime parameters (see `metadata.py`)
        
        ### returns
        + mmif object, with new app annotations.
        """
        
        # Run app on contained VideoDocument(s) in MMIF
        for videodocument in mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            # one view per video document
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            new_view.new_contain(AnnotationTypes.BoundingBox, document=videodocument.id, timeUnit=parameters["timeUnit"])
            self.logger.debug(f"Running on video {videodocument.location_path()}")
            mmif = self.run_on_video(mmif, videodocument, new_view, **parameters)

        # Run app on contained ImageDocument(s) in MMIF
        if mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            # one view for all image documents
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            new_view.new_contain(AnnotationTypes.BoundingBox)
            self.logger.debug(f"Running on all images")
            mmif = self.run_on_images(mmif, new_view)
        
        return mmif

    def run_on_images(self, mmif: Mmif, new_view: View) -> Mmif:
        """Run EAST on ImageDocuments

        ### params
        + mmif => Mmif Object 
        + new_view => a single mmif View (representing all ImageDocuments)

        ### returns
        + mmif, annotated with boundingboxes
        """
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

    def run_on_video(self, mmif: Mmif, videodocument: Document, new_view: View, **config) -> Mmif:
        """Run EAST on a VideoDocument

        ### params
        + mmif => Mmif Object 
        + videodocument => VideoDocument file
        + new_view => a single mmif View

        ### returns
        + mmif, annotated with boundingboxes
        """
        cap = vdh.capture(videodocument)
        views_with_tframe = [v for v in mmif.get_views_for_document(videodocument.id) 
                             if v.metadata.contains[AnnotationTypes.TimeFrame]]
        if views_with_tframe:
            frame_type = set(config["frameType"])
            frame_type.discard("")  # after this, if this set is empty, the next step will use "all" types
            # now for each of frame of interest, we will sample 2 (hard-coded) frames, evenly distributed
            target_frames = set()
            target_frames.update(*[np.linspace(*vdh.convert_timeframe(mmif, a, 'frame'), 2, dtype=int)
                                   for v in views_with_tframe for a in v.get_annotations(AnnotationTypes.TimeFrame)
                                   if not frame_type or a.get_property("frameType") in frame_type])
            target_frames = list(map(int, target_frames))
        else:
            target_frames = vdh.sample_frames(
                start_frame=0, 
                end_frame=min(int(config['stopAt']), videodocument.get_property("frameCount")),
                sample_rate=config['sampleRate']
            )

        target_frames.sort()

        for fn, fi in zip(target_frames, vdh.extract_frames_as_images(videodocument, copy.deepcopy(target_frames))):
            result_list = image_to_east_boxes(fi)
            for box in result_list:
                bb_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                tp = vdh.convert(t=fn, in_unit='frame', out_unit=config['timeUnit'], fps=videodocument.get_property("fps"))
                self.logger.debug(f"Adding a timepoint at frame: {fn} >> {tp}")

                tp_annotation = new_view.new_annotation(AnnotationTypes.TimePoint)
                tp_annotation.add_property("timeUnit", config["timeUnit"])
                tp_annotation.add_property("timePoint", tp)

                #bb_annotation.add_property("timePoint", tp)
                bb_annotation.add_property("boxType", "text")
                x0, y0, x1, y1 = box
                bb_annotation.add_property("coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]])

                alignment_annotation = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment_annotation.add_property("source", tp_annotation.id)
                alignment_annotation.add_property("target", bb_annotation.id)

        return mmif

def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any 
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this. 
    """
    return EastTextDetection()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen" )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = get_app()
    
    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
