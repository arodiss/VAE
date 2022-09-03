from tkinter import Canvas, Scale, HORIZONTAL, Tk, NW
from PIL import Image, ImageTk
from sklearn.datasets import fetch_olivetti_faces
from matplotlib import cm
import numpy as np


class FaceEditor:
    def __init__(self, face_array):
        self.root = Tk()
        self.root.title("Face editor")
        self.img = self.make_image(face_array)
        self.image_container = Canvas(self.root, borderwidth=5, width=300, height=300)
        self.starburst_slider = Scale(from_=0, to=100, orient=HORIZONTAL, command=self.starburst)
        self.tk_img = None

        self.image_container.pack()
        self.starburst_slider.pack()
        self.update_image()

        self.root.mainloop()

    def update_image(self):
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.image_container.create_image(0, 0, image=self.tk_img, anchor=NW)

    def starburst(self, var):
        self.img = self.make_image(face + float(var) * 0.01)
        self.update_image()

    @staticmethod
    def make_image(image_array):
        return Image.fromarray(np.uint8(cm.gray(image_array) * 255)).resize((300, 300))


faces = fetch_olivetti_faces()['images']
face = faces[0]
FaceEditor(face)
