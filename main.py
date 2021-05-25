from imutils.video import VideoStream
import cv2 as cv
import sys
import numpy as np
import os.path
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from collections import Counter
import time
import itertools
import os


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


cards = {
    'A_S': {'rank': 'Ace', 'suit': 'spades'},
    'J_S': {'rank': 'Jack', 'suit': 'spades'},
    'Q_S': {'rank': 'Queen', 'suit': 'spades'},
    'K_S': {'rank': 'King', 'suit': 'spades'},
    '9_S': {'rank': '9', 'suit': 'spades'},
    '10_S': {'rank': '10', 'suit': 'spades'},

    'A_H': {'rank': 'Ace', 'suit': 'hearths'},
    'J_H': {'rank': 'Jack', 'suit': 'hearths'},
    'Q_H': {'rank': 'Queen', 'suit': 'hearths'},
    'K_H': {'rank': 'King', 'suit': 'hearths'},
    '9_H': {'rank': '9', 'suit': 'hearths'},
    '10_H': {'rank': '10', 'suit': 'hearths'},

    'A_D': {'rank': 'Ace', 'suit': 'diamonds'},
    'J_D': {'rank': 'Jack', 'suit': 'diamonds'},
    'Q_D': {'rank': 'Queen', 'suit': 'diamonds'},
    'K_D': {'rank': 'King', 'suit': 'diamonds'},
    '9_D': {'rank': '9', 'suit': 'diamonds'},
    '10_D': {'rank': '10', 'suit': 'diamonds'},

    'A_C': {'rank': 'Ace', 'suit': 'clubs'},
    'J_C': {'rank': 'Jack', 'suit': 'clubs'},
    'Q_C': {'rank': 'Queen', 'suit': 'clubs'},
    'K_C': {'rank': 'King', 'suit': 'clubs'},
    '9_C': {'rank': '9', 'suit': 'clubs'},
    '10_C': {'rank': '10', 'suit': 'clubs'},

}


RANKS = {
    'Ace': 1,
    'King': 2,
    'Queen': 3,
    'Jack': 4,
    '10': 5,
    '9': 6
}


def straight_flush(list):

    sorted_list = sorted(list, key=lambda x: RANKS[x['rank']])

    suit = sorted_list[0]['suit']
    rank = sorted_list[0]['rank']

    for card in sorted_list[1:]:
        if card['suit'] != suit:
            return False
        if RANKS[card['rank']] - RANKS[rank] != 1:
            return False

        rank = card['rank']

    return True


def four_kind(list):

    sorted_list = sorted(list, key=lambda x: RANKS[x['rank']])

    rank = sorted_list[1]['rank']
    repeats = 0

    for card in sorted_list:
        if card['rank'] == rank:
            repeats += 1

    if repeats != 4:
        return False

    return True


def full_house(list):

    sorted_list = sorted(list, key=lambda x: RANKS[x['rank']])
    highest_card = sorted_list[0]
    lowest_card = sorted_list[4]

    highest_sum = sum(c['rank'] == highest_card['rank'] for c in sorted_list)
    lowest_sum = sum(c['rank'] == lowest_card['rank'] for c in sorted_list)

    if (highest_sum == 3 and lowest_sum == 2):
        return True
    elif (highest_sum == 2 and lowest_sum == 3):
        return True
    else:
        return False


def flush(list):

    sorted_list = sorted(list, key=lambda x: RANKS[x['rank']])

    suit = sorted_list[0]['suit']

    for card in sorted_list[1:]:
        if card['suit'] != suit:
            return False

    return True


def straight(list):

    sorted_list = sorted(list, key=lambda x: RANKS[x['rank']])

    rank = sorted_list[0]['rank']

    for card in sorted_list[1:]:
        if RANKS[card['rank']] - RANKS[rank] != 1:
            return False

        rank = card['rank']

    return True


def three_kind(list):

    sorted_list = sorted(list, key=lambda x: RANKS[x['rank']])

    rank = sorted_list[1]['rank']
    repeats = 0

    for card in sorted_list:
        if card['rank'] == rank:
            repeats += 1

    if repeats != 3:
        return False

    return True


def check(list):
    result = []
    if len(list) != 5:
        return
    for n, f in [('Straight Flush', straight_flush), ('Four of a Kind', four_kind), ('Full House', full_house), ('Flush', flush), ('Straight', straight), ('Three of a Kind', three_kind)]:
        result.append((n, f(list)))

    return result


def normalizeCardCollection(collection):

    flat_list = list(itertools.chain(*collection))
    sorted_by_occurences = Counter(flat_list).most_common(5)
    global cards
    figures = [cards[sorted_by_occurences[i][0]]
               for i in range(len(sorted_by_occurences))]
    return figures


class NN():
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.net = cv.dnn.readNetFromDarknet(self.config, self.weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


class CardRecognizer():
    def __init__(self, network):
        self.window_name = 'Real-time card detection'
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, 800, 600)
        self.card_collection = []
        self.max_card_history_length = 10
        self.network = network
        self.classes_file = './resources/coco.names'
        self.classes = None
        with open(self.classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.RANDOM_COLORS = np.random.uniform(
            0, 255, size=(len(self.classes), 3))
        self.confidence_threshold = 0.6
        self.nms_threshold = 0.6

    def lookForCards(self, camera_stream=WebcamVideoStream(src=0).start(), input_width=224, input_height=224):
        frame = camera_stream.read()
        blob = cv.dnn.blobFromImage(
            frame, 1/255, (input_width, input_height), [0, 0, 0], 1, crop=False)
        self.network.net.setInput(blob)
        outs = network.net.forward(self.getLayerNames(network.net))
        frame, collection = self.processOutput(
            frame, outs), self.card_collection
        cv.imshow(self.window_name, frame)
        return frame, collection

    def drawBoundingBox(self, frame, classId, conf, left, top, right, bottom):

        color = self.RANDOM_COLORS[classId]
        cv.rectangle(frame, (left, top), (right, bottom), color, 1)

        label = f'{conf:.2f}'

        if self.classes:
            label = f'{self.classes[classId]}:{label}'

        label_size, base_line = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(
            1.5*label_size[0]), top + base_line), (0, 0, 0), cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return(frame)

    def processOutput(self, frame, outs):
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        found_cards = list(
            (sorted(set([self.classes[class_ids[i]] for i in range(len(class_ids))]))))

        self.card_collection.append(found_cards)
        self.card_collection = self.card_collection[-self.max_card_history_length:]

        indices = cv.dnn.NMSBoxes(
            boxes, confidences,  self.confidence_threshold,  self.nms_threshold)
        for (k, v) in enumerate(indices):
            box = boxes[k]
            left, top, width, height = box
            frame = self.drawBoundingBox(
                frame, class_ids[k], confidences[k], left, top, left + width, top + height)

        return frame

    def getLayerNames(self, net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def main():
    t0 = time.time()

    while True:
        frame, card_collection = card_recognizer.lookForCards()
        card_collection = normalizeCardCollection(card_collection)
        result = check(card_collection)

        #t1 = time.time()

        clearConsole()
        # print(result)
       # t0 = time.time()
        if result:
            for i, (k, v) in enumerate(result):
                print(k, ": ", v)
                print()

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            break


network = NN(r'./resources/yolov3_testing.cfg',
             r'./resources/yolov3_training_last.weights')
card_recognizer = CardRecognizer(network)

if __name__ == '__main__':
    main()
