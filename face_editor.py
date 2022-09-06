from tkinter import Canvas, Scale, HORIZONTAL, Tk, NW
from PIL import Image, ImageTk
from sklearn.datasets import fetch_olivetti_faces
from matplotlib import cm
import numpy as np
from trained_model import get_best_model
import torch


class FaceEditor:
    def __init__(self, raw_face_array):
        self.change = [0, 0, 0, 0, 0, 0, 0]
        self.model = get_best_model()
        self.root = Tk()
        self.root.title("Face editor")
        self.bottleneck = self.model.encode(torch.from_numpy(
            np.expand_dims(np.expand_dims(raw_face_array, axis=0), axis=0)
        ))[0].detach()
        self.raw_image_container = Canvas(self.root, borderwidth=5, width=300, height=300)
        self.reconstructed_image_container = Canvas(self.root, borderwidth=5, width=300, height=300)
        self.slider0 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change0)
        self.slider1 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change1)
        self.slider2 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change2)
        self.slider3 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change3)
        self.slider4 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change4)
        self.slider5 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change5)
        self.slider6 = Scale(from_=-100, to=100, orient=HORIZONTAL, command=self.change6)

        raw_img = Image.fromarray(np.uint8(cm.gray(raw_face_array) * 255)).resize((300, 300))
        raw_tk_img = ImageTk.PhotoImage(raw_img)
        self.raw_image_container.create_image(0, 0, image=raw_tk_img, anchor=NW)
        self.reconstructed_tk_img = None
        self.raw_image_container.pack()
        self.reconstructed_image_container.pack()
        self.slider0.pack()
        self.slider1.pack()
        self.slider2.pack()
        self.slider3.pack()
        self.slider4.pack()
        self.slider5.pack()
        self.slider6.pack()
        self.update_image()

        self.root.mainloop()

    def update_image(self):
        bottleneck = np.copy(self.bottleneck)
        bottleneck = bottleneck + np.array(self.change).astype(np.float32)
        reconstructed_face_array = self.model.decode(torch.from_numpy(
            bottleneck
        )).detach()[0][0]
        img = Image.fromarray(np.uint8(cm.gray(reconstructed_face_array) * 255)).resize((300, 300))
        self.reconstructed_tk_img = ImageTk.PhotoImage(img)
        self.reconstructed_image_container.create_image(0, 0, image=self.reconstructed_tk_img, anchor=NW)

    def change0(self, var):
        self.change[0] = float(var) * 0.1
        self.update_image()

    def change1(self, var):
        self.change[1] = float(var) * 0.1
        self.update_image()

    def change2(self, var):
        self.change[2] = float(var) * 0.1
        self.update_image()

    def change3(self, var):
        self.change[3] = float(var) * 0.1
        self.update_image()

    def change4(self, var):
        self.change[4] = float(var) * 0.1
        self.update_image()

    def change5(self, var):
        self.change[5] = float(var) * 0.1
        self.update_image()
    def change6(self, var):
        self.change[6] = float(var) * 0.1
        self.update_image()


FaceEditor(fetch_olivetti_faces()['images'][0])
