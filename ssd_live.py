import base64
import sys
from multiprocessing import Queue
import threading
import uuid
import time
import json

import cv2
import easyocr
import numpy as np
from paddleocr import PaddleOCR
import paho.mqtt.client as mqtt

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from MainWindow import Ui_MainWindow

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer

pre = ["", "", "", "", ""]
nxt = ""
shareData = Queue(10)
shareVideo = Queue(5000)
storeVideo = Queue(5000)
mqttMsg = Queue(10)


def init():
    model_path = "models/mb1.pth"
    label_path = "models/open-images-model-labels.txt"

    # cap = cv2.VideoCapture('rtmp://192.168.1.199:1935/myapp/22')   # capture from camera
    capture = cv2.VideoCapture(r"1.mp4")
    # capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(3, 1280)
    capture.set(4, 720)

    classNames = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(classNames)

    net = create_mobilenetv1_ssd(len(classNames), is_test=True)
    net.load(model_path)
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    exitFlag = 0

    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en")
    return capture, classNames, predictor, ocr


def mqttConnect():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set("test", "123456")
    client.connect('mqtt.drivethru.top', 1883, 600)
    return client


def on_connect(client, userdata, flags, rc):
    print("Connected with result code: " + str(rc))


def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


class myThread(threading.Thread):
    def __init__(self, threadID, name, ocr, dst, orig_image, client, data, previous):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.ocr = ocr
        self.dst = dst
        self.orig_image = orig_image
        self.client = client
        self.data = data
        self.previous = previous
        self.result = None

    def run(self):
        self.result = detectImg(self.ocr, self.dst, self.orig_image, self.client, self.data, self.previous)

    def get_result(self):
        return self.result


def detection(capture, classNames, predictor, ocr, client):
    global pre
    global shareData
    timer = Timer()
    count = 0
    detResult = []
    confidence = ""

    # while capture.isOpened():
    while True:
        ret, orig_image = capture.read()
        if orig_image is None:
            continue
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        timer.start()
        boxes, labels, probabilities = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        # print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        maxPredication = 0.0
        maxIndex = -1
        for i in range(boxes.size(0)):
            if probabilities[i] > maxPredication:
                maxPredication = probabilities[i]
                maxIndex = i
            box = boxes[i, :]
            label = f"{classNames[labels[i]]}: {probabilities[i]:.2f}"
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (int(box[0]) + 20, int(box[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        if maxIndex > -1:
            count = 0
            cutBox = boxes[maxIndex, :]
            dst = img[int(cutBox[1]):int(cutBox[3]), int(cutBox[0]):int(cutBox[2])]

            suid = ''.join(str(uuid.uuid4()).split('-'))
            timestamp = int(time.time())
            confidence = f'{maxPredication: .2f}'
            data = {"carPlate": "", "confidence": confidence, "suid": suid, "timestamp": timestamp}

            thread1 = myThread(1, "Thread-1", ocr, dst, orig_image, client, data, pre)
            thread1.start()
            thread1.join()

            print("NMSL" + label)
            print("NMSL" + confidence)
            result = thread1.get_result()
            if result:
                detText = result[0]
                isStore = result[1]
                pre.pop(0)
                pre.append(detText)
                if isStore is not None and isStore == "1":
                    shareData.put([orig_image, suid])
            # saveImg(orig_image, data)
        shareVideo.put(orig_image)
        storeVideo.put(orig_image)
        # orig_image = cv2.resize(orig_image, None, fx=0.6, fy=1.4)
        # cv2.imshow('annotated', orig_image)
        # print('maxIndex: ',maxIndex)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def detectImg(ocr, dst, orig_image, client, data, previous):
    returnValue = []
    isStore = "0"
    global shareData
    timer = Timer()
    timer.start()
    text = ""
    # reader = easyocr.Reader(['en'])
    # result = reader.readtext(dst)
    # result
    # try:
    #    text = result[0][-2]
    #    print('car:', text)
    # except Exception as r:
    #    print('error%s' % r)
    try:
        result = ocr.ocr(dst, cls=True)
        for t in result:
            text = text + t[1][0]
    except Exception as r:
        print('error%s' % r)
    text = str(text)
    text = text.replace(" ", "")
    text = text.upper()
    interval = timer.end()
    print("Interval: " + str(interval))
    print(text)

    if text != "":
        if checkLoop(previous, text):
            isStore = "1"
            # cv2.imshow('detectedC:', dst)
            data["carPlate"] = text
            payload = json.dumps(data)
            bytePayload = bytes(payload, encoding='UTF-8')
            encodedBytePayload = base64.encodebytes(bytePayload)
            print("Sending Out")
            print(payload, encodedBytePayload)

            while True:
                msg = encodedBytePayload
                client.publish('search', payload=msg, qos=0, retain=False)
                break
        returnValue.append(text)
        returnValue.append(isStore)
        return returnValue


def checkLoop(previous, text):
    isExisting = True
    for i in range(1, len(previous)):
        if text == previous[i]:
            isExisting = False
    return isExisting


class threadDetection(threading.Thread):
    def run(self):
        cap, class_names, predict, paddle_ocr = init()
        mqtt_client = mqttConnect()
        detection(cap, class_names, predict, paddle_ocr, mqtt_client)


class threadSaveImg(threading.Thread):
    def run(self):
        global shareData
        while True:
            saveImage, suid = shareData.get()
            if suid:  # and data is not None:
                print("Saved")
                filename = suid + '.jpg'
                save_path = ".//images/"
                image = np.ascontiguousarray(saveImage)
                if image != "":
                    cv2.imwrite(save_path + filename, saveImage, [cv2.IMWRITE_JPEG_QUALITY, 75])
            time.sleep(2)


class threadPrint(threading.Thread):
    def run(self):
        global shareVideo
        while True:
            frame = shareVideo.get()
            frame = cv2.resize(frame, None, fx=0.65, fy=0.65)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()


class threadUI(threading.Thread):
    def run(self):
        app = QApplication(sys.argv)
        # create and show mainWindow
        mainWindow = MainWindow()
        mainWindow.show()
        sys.exit(app.exec_())


class threadGetMessage(threading.Thread):
    def run(self):
        global mqttMsg
        mqtt_client = mqttConnect()
        subscribe(mqtt_client)
        mqtt_client.loop_forever()


def on_message(client, userdata, msg):
    json_obj = json.loads(msg.payload.decode())
    mqttMsg.put(json_obj)
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")


def subscribe(client):
    client.subscribe("result")
    client.on_message = on_message


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        global storeVideo
        global mqttMsg
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.model = QStandardItemModel(0, 2)
        self.detResult = []
        self.tableViewInit()
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.pushButton.clicked.connect(self.controlTimer)

        # start/stop timer

    def tableViewInit(self):
        self.model.setHorizontalHeaderLabels(['CarPlate No.', 'Class', 'Name'])
        self.tableView.setModel(self.model)
        self.tableView.setColumnWidth(0, 130)
        self.tableView.setColumnWidth(1, 70)
        self.tableView.setColumnWidth(2, 300)
        self.tableView.horizontalHeader().setFont(QFont("Verdana", 13, QFont.Bold))
        '''
        item1 = QStandardItem('%s' % '')
        self.model.setItem(0, 0, item1)
        item2 = QStandardItem('%s' % '')
        self.model.setItem(0, 1, item2)
        item3 = QStandardItem('%s' % '')
        self.model.setItem(0, 2, item3)
        self.tableView.setModel(self.model)
        '''
    def dataChange(self):
        word = "Cars that have arrived"
        cursor = self.textBrowser.textCursor()
        cursor.insertHtml('''<p><span style="color: red; font-weight: bold; font-size: 20px">{} </span>'''.format(word))
        self.textBrowser.setText(word)

    def columnInsert(self):
        if mqttMsg.qsize() > 0:
            i = mqttMsg.get()
            item1 = QStandardItem('%s' % i["carPlate"])
            # self.model.setItem(i, 0, item1)
            item2 = QStandardItem('%s' % i["studentClass"])
            # self.model.setItem(i, 1, item2)
            item3 = QStandardItem('%s' % i["studentName"])
            # self.model.setItem(i, 2, item3)
            self.model.appendRow([item1, item2, item3])
            self.tableView.setModel(self.model)

    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            self.dataChange()
            # create video capture
            # self.cap = cv2.VideoCapture(r"..//1.mp4")
            # self.cap.set(3, 1280)
            # self.cap.set(4, 720)
            # start timer
            # self.image = self.ssd_live.getData()
            self.timer.start(2)
            # update control_bt text
            self.pushButton.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # update control_bt text
            self.label_image.setText("Stop View")
            self.textBrowser.setText("")
            self.pushButton.setText("Start")

    # view camera
    def viewCam(self):
        # read image in BGR format
        # ret, image = self.cap.read()
        # convert image to RGB format
        # image = self.image
        image = storeVideo.get()
        image = cv2.resize(image, None, fx=0.65, fy=0.65)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.label_image.setPixmap(QPixmap.fromImage(qImg))
        self.columnInsert()


if __name__ == '__main__':
    t1 = threadDetection()
    t3 = threadPrint()
    t2 = threadSaveImg()
    t4 = threadUI()
    t5 = threadGetMessage()

    t1.start()
    t3.start()
    t2.start()
    t4.start()
    t5.start()


def start():
    t1.start()
    t3.start()
    t2.start()


def stop():
    t1.join()
    t3.join()
    t2.join()
    # t4.join()
