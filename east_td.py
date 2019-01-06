import imutils
import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from clams.serve import ClamApp
from clams.serialize import *
from clams.vocab import AnnotationTypes
from clams.vocab import MediaTypes
from clams.restify import Restifier


class EAST_td(ClamApp):

    def appmetadata(self):
        metadata = {"name": "EAST Text Detection",
                    "description": "This tool applies EAST test detection to the video.",
                    "vendor": "Team CLAMS",
                    "requires": [MediaTypes.V],
                    "produces": [AnnotationTypes.BOX]}
        return metadata

    def sniff(self, mmif):
        # this mock-up method always returns true
        return True

    def annotate(self, mmif_json):
        mmif = Mmif(mmif_json)
        video_filename = mmif.get_medium_location(MediaTypes.V)
        east_output = self.run_EAST(video_filename, mmif_json) #east_output is a list of frame number, [(x1, y1, x2, y2)] pairs

        new_view = mmif.new_view()
        contain = new_view.new_contain(AnnotationTypes.OCR)
        contain.producer = self.__class__

        for int_id, (start_frame, box_list) in enumerate(east_output):
            annotation = new_view.new_annotation(int_id)
            annotation.start = str(start_frame)
            annotation.end = str(start_frame)  # since we're treating each frame individually for now, start and end are the same, eventually we'll want some kind of coreference resolution for boxes across multiple frames
            annotation.feature = {'boxes':box_list}
            annotation.attype = AnnotationTypes.TBOX

        for contain in new_view.contains.keys():
            mmif.contains.update({contain: new_view.id})
        return mmif

    @staticmethod
    def run_EAST(video_filename, mmif): # mmif here will be used for filtering out frames/
        #apply tesseract ocr to frames
        sample_ratio = 60
        box_min_conf = .5 #minimum acceptable confidence

        def process_image(f):
            proc = cv2.medianBlur(f, 5) # reduce noise
            return proc

        def decode_predictions(scores, geometry):
            '''
            Taken from pyimagesearch, convert results to rectangles and confidences
            '''

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
            return (rects, confidences)

        # initialize the original frame dimensions, new frame dimensions,
        # and ratio between the dimensions
        (W, H) = (None, None)
        (newW, newH) = (320, 320)  # newH and newW must a multiple of 32.
        (rW, rH) = (None, None)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

        net = cv2.dnn.readNet(os.path.join(".","frozen_east_text_detection.pb")) # load the model
        cap = cv2.VideoCapture(video_filename)

        counter = 0
        result = []
        while cap.isOpened():
            ret, f = cap.read()

            if not ret:
                break
            if counter % sample_ratio == 0:
                # resize the frame, maintaining the aspect ratio
                f = imutils.resize(f, width=900)
                processed = process_image(f)
                orig = f.copy()

                # if our frame dimensions are None, we still need to compute the
                # ratio of old frame dimensions to new frame dimensions
                if W is None or H is None:
                    (H, W) = f.shape[:2]
                    rW = W / float(newW)
                    rH = H / float(newH)

                # resize the frame, this time ignoring aspect ratio
                f = cv2.resize(f, (newW, newH))

                # construct a blob from the frame and then perform a forward pass
                # of the model to obtain the two output layer sets

                blob = cv2.dnn.blobFromImage(processed, 1.0, (newW, newH),
                                                       (123.68, 116.78, 103.94), swapRB=True, crop=False)

                net.setInput(blob)
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
            if len(box_list) > 0:
                result.append((counter, box_list))
            counter += 1
        return result



if __name__ == "__main__":
    td_tool = EAST_td()
    td_service = Restifier(td_tool)
    td_service.run()

