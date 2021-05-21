from imutils.video import VideoStream
import cv2 as cv
import sys
import numpy as np
import os.path
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from collections import Counter

class NN():
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.net = cv.dnn.readNetFromDarknet(self.config, self.weights )
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


class CardRecognizer():
    def __init__(self, network ):
        self.window_name = 'Real-time card detection'
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, 800, 600)
        self.card_collection = []
        self.max_card_history_length = 10
        self.network = network
        self.classes_file = "coco.names"
        self.classes = None
        with open(self.classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.RANDOM_COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.confidence_threshold = 0.9
        self.nms_threshold = 0.6

    
    def lookForCards(self, camera_stream=WebcamVideoStream(src=0).start(), input_width=224, input_height=224):
            frame = camera_stream.read()
            blob = cv.dnn.blobFromImage(frame, 1/255, (input_width, input_height), [0,0,0], 1, crop=False)
            self.network.net.setInput(blob)
            outs = network.net.forward(self.getLayerNames(network.net))
            frame, collection = self.processOutput(frame, outs), self.card_collection
            cv.imshow(self.window_name, frame)
            return frame, collection

    def drawBoundingBox(self, frame, classId, conf, left, top, right, bottom):
        
        color = self.RANDOM_COLORS[classId]
        cv.rectangle(frame, (left, top), (right, bottom), color, 1)

        label = f'{conf:.2f}'
        
        if  self.classes:
            label = f'{self.classes[classId]}:{label}'

        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top + base_line), (0, 0, 0), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return(frame)

    def processOutput(self, frame, outs):
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:] #WHY 5?
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence >  self.confidence_threshold:
                    #calculating position
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        found_cards = dict(Counter(sorted(list([self.classes[class_ids[i]] for i in range(len(class_ids))]))))
        normalized_card_count = { k : np.ceil(v/2) for _, (k,v) in enumerate(found_cards.items())}
        self.card_collection.append(normalized_card_count)
        self.card_collection = self.card_collection[-self.max_card_history_length:]


        indices = cv.dnn.NMSBoxes(boxes, confidences,  self.confidence_threshold,  self.nms_threshold)
        for (k,v) in enumerate(indices):
            box = boxes[k]
            left, top, width, height =  box
            frame = self.drawBoundingBox(frame, class_ids[k], confidences[k], left, top, left + width, top + height)

        return frame

    def getLayerNames(self, net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

   
def main():

    while True:
        frame, card_collection = card_recognizer.lookForCards()
        print(card_collection)
        
    

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            cv.destroyAllWindows()
            break

network = NN(r"yolov3_testing.cfg", r"yolov3_training_last.weights")
card_recognizer = CardRecognizer(network)

if __name__ == '__main__':
    main()