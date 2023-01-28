import base64
import io
import requests
from typing import List, Optional, Union
from PIL import Image
from PySide6.QtCore import Qt, QByteArray, QBuffer, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QLabel, QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QWidget
import json
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PySide6.QtCore import QUrl
from msgpack import unpackb
from PIL import Image
import cv2

class InferenceArgs(dict):
    def __init__(self):
        super().__init__()
        self.prompt = ["Portrait of a gorgeous princess, by WLOP, Stanley Artgerm Lau, trending on ArtStation"]
        self.negative_prompt = [
            "Horrible, very ugly, jpeg artifacts, messy, warped, split, bad anatomy, malformed body, malformed, warped, fake, 3d, drawn, hideous, disgusting"]
        self.height = 512
        self.width = 512
        self.guidance_scale = 8.0
        self.seed = None
        self.num_inference_steps = 50


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.label.setScaledContents(True)
        self.prompt_input = QLineEdit()
        self.negative_prompt_input = QLineEdit()
        self.height_input = QLineEdit()
        self.width_input = QLineEdit()
        self.guidance_scale_input = QLineEdit()
        self.seed_input = QLineEdit()
        self.num_inference_steps_input = QLineEdit()

        self.prompt_input.setText("Prompt")
        self.height_input.setText("512")
        self.width_input.setText("512")
        self.guidance_scale_input.setText("7.5")
        self.seed_input.setText("100000")
        self.num_inference_steps_input.setText("20")

        self.strength = QDoubleSpinBox()
        self.strength.setMinimum(0.00)
        self.strength.setMaximum(2.00)
        self.strength.setSingleStep(0.01)
        self.strength.setValue(0.80)


        self.infer_button = QPushButton("Infer")
        self.infer_button.clicked.connect(self.on_infer_button_clicked)
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.handle_response)
        layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.prompt_input)
        h_layout.addWidget(self.negative_prompt_input)
        layout.addLayout(h_layout)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.height_input)
        h_layout.addWidget(self.width_input)
        layout.addLayout(h_layout)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.guidance_scale_input)
        h_layout.addWidget(self.seed_input)
        h_layout.addWidget(self.num_inference_steps_input)
        layout.addLayout(h_layout)
        layout.addWidget(self.infer_button)
        layout.addWidget(self.label)
        layout.addWidget(self.strength)

        self.setLayout(layout)
        self.busy = False
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def on_infer_button_clicked(self):

        request = QNetworkRequest(QUrl("http://172.17.0.2:5003/api/infer"))
        body = InferenceArgs()
        body["prompt"] = self.prompt_input.text().split(',')
        body["negative_prompt"] = self.negative_prompt_input.text().split(',')
        body["height"] = int(self.height_input.text())
        body["width"] = int(self.width_input.text())
        body["num_inference_steps"] = int(self.num_inference_steps_input.text())
        body["seed"] = self.seed_input.text() if self.seed_input.text() != "" else None
        body["guidance_scale"] = float(self.guidance_scale_input.text())
        body["strength"] = float(self.strength.value())
        _, frame = self.video_capture.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer.tobytes()).decode()
        # Add image to request body
        body["webcam_image"] = image_base64

        request_data = QByteArray()
        request_data.append(bytes(json.dumps(body), 'utf-8'))
        request.setHeader(QNetworkRequest.ContentTypeHeader, 'application/json')
        request.setHeader(QNetworkRequest.ContentLengthHeader, request_data.size())
        self.network_manager.post(request, request_data)



        """args = InferenceArgs()
        args.prompt = self.prompt_input.text()
        args.negative_prompt = self.negative_prompt_input.text().split(",")
        args.height = int(self.height_input.text())
        args.width = int(self.width_input.text())
        args.guidance_scale = float

        args.guidance_scale = float(self.guidance_scale_input.text())
        args.seed = int(self.seed_input.text()) if self.seed_input.text() else None
        args.num_inference_steps = int(self.num_inference_steps_input.text())

        json_data = json.dumps(args).encode()
        print(json_data)
        self.request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        self.network_manager.post(self.request, json_data)"""

    def handle_response(self, reply):
        self.busy = False
        #print(reply)
        data = reply.readAll().data()
        result = unpackb(data)
        result['images'] = [Image.open(
            io.BytesIO(i)
        ) for i in result['images']]
        print(result['images'])
        result['images'][0].save("result.png")
        #print(data)
        #json_response = json.loads(reply.readAll().data().decode())
        #image_data = Image.frombytes(data)
        self.img = QPixmap("result.png")

        #self.img.loadFromData(data)

        self.label.setPixmap(self.img)
        #self.timer = QTimer()
        #self.timer.setSingleShot(True)
        #self.timer.setInterval(5)
        #self.timer.timeout.connect(self.on_infer_button_clicked)
        #self.timer.start()
        self.on_infer_button_clicked()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
