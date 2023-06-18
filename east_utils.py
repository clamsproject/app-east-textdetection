from typing import List, Tuple

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from mmif import Mmif, View, DocumentTypes, AnnotationTypes

BOX_MIN_CONF = 0.1
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


def decode_predictions(scores, geometry, box_min_conf=BOX_MIN_CONF):
    """
    Taken from pyimagesearch, convert results to rectangles and confidences
    """

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < box_min_conf:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


def image_to_east_boxes(image: np.array) -> List[Tuple[int, int, int, int]]:
    (newW, newH) = (320, 320)  # newH and newW must a multiple of 32.
    (H, W) = image.shape[:2]
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the frame, this time ignoring aspect ratio
    image = cv2.resize(image, (newW, newH))

    # construct a blob from the frame and then perform a forward pass
    # of the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    box_list = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        box_list.append((startX, startY, endX, endY))
    return box_list


def get_target_frame_numbers(mmif, frame_type, frames_per_segment=2):
    def convert_msec(time_msec):
        import math
        return math.floor(time_msec * 29.97)  # todo 6/1/21 kelleylynch assuming frame rate

    views_with_tframe = [
        tf_view
        for tf_view in mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
        if tf_view.get_annotations(AnnotationTypes.TimeFrame, frameType=frame_type)
    ]
    frame_number_ranges = [
        (tf_annotation.properties["start"], tf_annotation.properties["end"])
        if tf_view.metadata.get_parameter("timeUnit") in ["frames", "frame"]
        else (convert_msec(tf_annotation.properties["start"]), convert_msec(tf_annotation.properties["end"]))
        for tf_view in views_with_tframe
        for tf_annotation in tf_view.get_annotations(AnnotationTypes.TimeFrame, frameType=frame_type)
    ]
    target_frames = list(set([int(f) for start, end in frame_number_ranges
                              for f in np.linspace(start, end, frames_per_segment, dtype=int)]))

    return target_frames


def boxes_from_target_frames(target_frames:List[int], cap:cv2.VideoCapture, new_view:View):
    for frame_number in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, f = cap.read()
        result_list = image_to_east_boxes(f)
        for box in result_list:
            bb_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
            bb_annotation.add_property("boxType", "text")
            x0, y0, x1, y1 = box
            bb_annotation.add_property(
                "coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
            )
            bb_annotation.add_property("frame", frame_number)
    

def run_EAST_video(mmif: Mmif, new_view: View, **kwargs) -> Mmif:
    cap = cv2.VideoCapture(mmif.get_document_location(DocumentTypes.VideoDocument))
    counter = 0
    idx = 0
    stop_at = int(kwargs["stopAt"])
    if "frameType" in kwargs:
        frame_type = kwargs["frameType"]
    else:
        frame_type = ""
    target_frames = []
    if frame_type:
        target_frames = get_target_frame_numbers(mmif, frame_type, 2)
        boxes_from_target_frames(target_frames, cap, new_view)
    else:
        while cap.isOpened():
            if counter > stop_at:
                break
            ret, f = cap.read()
            if target_frames:
                if counter not in target_frames:
                    counter += 1 #todo move this
                    continue
            if not ret:
                break
            if (counter % kwargs['sampleRatio'] == 0) or (counter in target_frames):
                result_list = image_to_east_boxes(f)
                for box in result_list:
                    idx += 1
                    bb_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                    bb_annotation.add_property("boxType", "text")
                    x0, y0, x1, y1 = box
                    bb_annotation.add_property(
                        "coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
                    )
                    bb_annotation.add_property("frame", counter)
            counter += 1
    return mmif


def run_EAST_image(mmif: Mmif, new_view:View) -> Mmif:
    image = cv2.imread(mmif.get_document_location(DocumentTypes.ImageDocument))
    box_list = image_to_east_boxes(image)
    for idx, box in enumerate(box_list):
        annotation = new_view.new_annotation(f"td{idx}", AnnotationTypes.BoundingBox)
        annotation.add_property("boxType", "text")
        x0, y0, x1, y1 = box
        annotation.add_property(
            "coordinates", [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
        )
    return mmif
