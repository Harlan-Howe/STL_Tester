import numpy as np
import cv2
from STL_Object_File import STL_Object
from tkinter import Tk, Label, filedialog


class STL_Visualizer:

    def __init__(self, filename):
        self.object: STL_Object = STL_Object(filename)
        self.transform: np.ndarray = np.eye(4, dtype=float)

    def display(self):
        screen: np.ndarray = np.ones((400, 400), dtype=float)

        self.object.draw_self(screen, self.transform)

        cv2.imshow("Screen", screen)
        cv2.waitKey(1)


def get_file_and_visualize():
    root = Tk()
    Label(root, text="Showing file dialog").pack()
    root.update()
    filename: str = filedialog.askopenfilename(title="Find the stl file to display")
    root.withdraw()

    visualizer = STL_Visualizer(filename)
    visualizer.display()

    print("Press any key to end program.")
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_file_and_visualize()
