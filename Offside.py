from ultralytics import YOLO
from tkinter import messagebox
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tkinter as tk
import threading


class OffsideClass():

    def __init__(self, image_path):
        self.__image_path = image_path
        self.__image = cv.imread(image_path)
        self.__gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        self.__delete_lines = []
        self.__final_lines = []
        self.__line = 2
        self.__vanishing_point = (0, 0)


    def static_line_detection(self, max_width=800):

        def resize_with_aspect_ratio(image, width=None, height=None):
                h, w = image.shape[:2]
                
                if width is None and height is None:
                    return image
                
                if width is not None:
                    scale = width / w
                    new_width, new_height = width, int(h * scale)
                else:
                    scale = height / h
                    new_width, new_height = int(w * scale), height

                return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    

        def select_points(event, x, y, flags, param):
            clone = image.copy()
            clone = resize_with_aspect_ratio(clone, max_width)

            if event == cv.EVENT_LBUTTONDOWN:
                points.append((x, y))

                if len(points)%2 == 0:
                    # draw points
                    cv.circle(clone, points[0], 5, (0, 0, 255), -1)
                    cv.circle(clone, points[1], 5, (0, 0, 255), -1)
                    # draw lines
                    cv.line(clone, points[0], points[1], (0, 255, 0), 5)

                    if len(points) == 4:
                        # draw points
                        cv.circle(clone, points[2], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[3], 5, (0, 0, 255), -1)
                        # draw lines
                        cv.line(clone, points[2], points[3], (0, 255, 0), 5)

                elif len(points)%2 == 1:

                    if len(points) == 1:
                        cv.circle(clone, points[-1], 5, (0, 0, 255), -1)

                    elif len(points) == 3:
                        # draw points
                        cv.circle(clone, points[-1], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[0], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[1], 5, (0, 0, 255), -1)
                        # draw lines
                        cv.line(clone, points[0], points[1], (0, 255, 0), 5)

                    elif len(points) == 5:
                        # draw points
                        cv.circle(clone, points[-1], 5, (0, 0, 255), -1)
                        # draw lines
                        cv.line(clone, points[0], points[1], (0, 255, 0), 5)
                        cv.line(clone, points[2], points[3], (0, 255, 0), 5)

                cv.imshow("Select Lines", clone)

            
            elif event == cv.EVENT_RBUTTONDOWN:
                try:
                    points.remove(points[-1])
                except:
                    print('ERROR: There is no point.')

                if len(points)%2 == 0:
                    if len(points) == 0:
                        clone = image.copy()

                    elif len(points) == 2:
                        # draw points
                        cv.circle(clone, points[0], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[1], 5, (0, 0, 255), -1)
                        # draw lines
                        cv.line(clone, points[0], points[1], (0, 255, 0), 5)

                    elif len(points) == 4:
                        # draw points
                        cv.circle(clone, points[0], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[1], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[2], 5, (0, 0, 255), -1)
                        cv.circle(clone, points[3], 5, (0, 0, 255), -1)
                        # draw lines
                        cv.line(clone, points[0], points[1], (0, 255, 0), 5)
                        cv.line(clone, points[2], points[3], (0, 255, 0), 5)

                elif len(points)%2 == 1:

                    if len(points) == 1:
                        cv.circle(clone, points[-1], 5, (0, 0, 255), -1)

                    elif len(points) == 3:
                        cv.circle(clone, points[-1], 5, (0, 0, 255), -1)
                        cv.line(clone, points[0], points[1], (0, 255, 0), 5)

                
                cv.imshow("Select Lines", clone)
                  
                    
        points = []
        image = resize_with_aspect_ratio(cv.imread(self.__image_path), max_width)
        cv.imshow("Select Lines", image)
        
        while True:
            cv.setMouseCallback("Select Lines", select_points)
            key = cv.waitKey(1) & 0xFF

            if len(points) == 5:
                cv.destroyAllWindows()
                break

            elif key == 27:
                cv.destroyAllWindows()
                break

            elif key == ord('c'):
                clone = image.copy()
                points = []
                cv.imshow("Select Lines", image)


        points = [[*points[0],*points[1]], [*points[2],*points[3]]]
        return (points)


    def player_detection(self, show):
        # use yolo pretrain model
        model = YOLO("yolov8n.pt")

        # path of image
        self.__image = cv.cvtColor(self.__image, cv.COLOR_BGR2RGB)

        # run model on image
        results = model(self.__image)

        # show human boxes
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]  # Detection percentage
                cls = int(box.cls[0])

                if conf > 0.4:  # Only objects with confidence above 40%
                    cv.rectangle(self.__image, (x1, y1), (x2, y2), (0, 0, 0), 1)
                    cv.putText(self.__image, f"Class: {cls}", (x1, y1 - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # show final detection image
        if show:
            plt.imshow(self.__image)
            plt.axis('off')
            plt.show()


    def increase_contrast(self, show):
        # increasing contrast
        self.equalized_image = cv.equalizeHist(self.__gray_image)

        # show image
        if show:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(self.__image, cmap='gray')
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(self.equalized_image, cmap='gray')
            plt.title("Contrast Enhanced Image")
            plt.axis("off")

            plt.show()


    def find_vanishing_point(self, line1, line2):
        # extracr points from lines
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # calculate the equation of the line
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3

        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            return None  # lines are parallel

        # calculate the collision point
        x = (C1 * B2 - C2 * B1) / determinant
        y = (A1 * C2 - A2 * C1) / determinant

        self.__vanishing_point = (int(x), int(y))
        return self.__vanishing_point


    def auto_line_detection(self, max_width=500):
        # detect farallel lines with trackbar values
        def detect_farallel_lines(max_width, max_Height=None):

            def resize_with_aspect_ratio(image, width=None, height=None):
                h, w = image.shape[:2]
                
                if width is None and height is None:
                    return image
                if width is not None:
                    scale = width / w
                    new_width, new_height = width, int(h * scale)
                else:
                    scale = height / h
                    new_width, new_height = int(w * scale), height

                return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)


            def adjust_threshold(x=None):
                clone = image.copy()
                # get trackbar value
                try:
                    low_thresh = cv.getTrackbarPos('Low', 'image')
                    high_thresh = cv.getTrackbarPos('High', 'image')
                    on_thresh = cv.getTrackbarPos('Off/On', 'image')
                    rdn = cv.getTrackbarPos('angle', 'image')
                    save = cv.getTrackbarPos('save', 'image')
                except:
                    return

                # change image size
                resized_original = resize_with_aspect_ratio(image, width=max_width)

                # use equal image if not None else use gray scale image
                try:
                    gray = self.equalized_image
                except:
                    gray = self.__gray_image
                
                # line detection
                edges = cv.Canny(gray, low_thresh, high_thresh)
                lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
                height, width = image.shape[:2]

                if lines is not None:
                    line_counter = 1  # lines counting
                    for line in lines:
                        x1, y1, x2, y2 = line[0]

                        # define line angle
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        
                        # select lines in special range
                        if rdn > abs(angle) > rdn - 20:
                            if line_counter not in self.__delete_lines:

                                if x2 != x1: # not error handling
                                    slope = (y2 - y1) / (x2 - x1)
                                    intercept = y1 - slope * x1

                                    # continuation of lines
                                    x1_new = 0
                                    y1_new = int(slope * x1_new + intercept)
                                    x2_new = width
                                    y2_new = int(slope * x2_new + intercept)

                                    # draw continuation lines
                                    cv.line(clone, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)

                                    # write line number in image
                                    cv.putText(clone, str(line_counter), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                    # if save is True create finel lines list
                                    if save:
                                        self.__final_lines.append([x1_new, y1_new, x2_new, y2_new])
                                        
                            line_counter += 1


                # if trackbars is on
                if on_thresh:
                    resized = resize_with_aspect_ratio(clone,  width=max_width)
                    # show image
                    cv.imshow('image', resized)

                    # save final lines
                    if save:
                        cv.destroyAllWindows()
                        if len(self.__final_lines) == 2:
                            line1, line2 = self.__final_lines
                        else:
                            print("more than len 2")
                
                # show original image if trackbars is off
                else:
                    img = cv.imread(self.__image_path)
                    resized_original = resize_with_aspect_ratio(img, width=max_width)
                    cv.imshow('image', resized_original)

            
            # read image and error handling
            image = cv.imread(self.__image_path)
            if image is None:
                print("❌ Error: Image not found! Make sure the file exists in the correct path.")
                exit()

            # create window
            cv.namedWindow('image')

            # short delay
            cv.waitKey(1)

            # create trackbars
            cv.createTrackbar('Low', 'image', 50, 255, adjust_threshold)
            cv.createTrackbar('High', 'image', 150, 255, adjust_threshold)
            cv.createTrackbar('angle', 'image', 0, 360, adjust_threshold)
            cv.createTrackbar('save', 'image', 0, 1, adjust_threshold)
            cv.createTrackbar('Off/On', 'image', 0, 1, adjust_threshold)

            # short delay
            cv.waitKey(100)  

            # first show detection lines
            adjust_threshold()

            # close window
            cv.waitKey(0)
            cv.destroyAllWindows()

        # add number to list
        def add_number():
            num = entry.get().strip()
            if num.isdigit():
                self.__delete_lines.append(int(num))
                entry.delete(0, tk.END)
            else:
                messagebox.showerror("ERROR", "You can enter number only")

        # run Tkinter window
        def run_tkinter():
            root = tk.Tk()
            root.title("Enter number of line")
            root.geometry("300x200")

            global entry
            entry = tk.Entry(root, font=("Arial", 14))
            entry.pack(pady=10)

            btn_add = tk.Button(root, text="Add", command=add_number, bg="green", fg="white", font=("Arial", 14))
            btn_add.pack(pady=5)

            root.mainloop()

        # run OpenCV window
        def run_opencv():
            detect_farallel_lines(max_width)

        # running both parts in separate threads
        thread_tk = threading.Thread(target=run_tkinter)
        thread_cv = threading.Thread(target=run_opencv)

        thread_tk.start()
        thread_cv.start()

        thread_tk.join()
        thread_cv.join()

        # return final lines list
        return (self.__final_lines)


    def draw_line(self, vanishing_point, max_width=800):

        def resize_with_aspect_ratio(image, width=None, height=None):
                h, w = image.shape[:2]
                
                if width is None and height is None:
                    return image
                if width is not None:
                    scale = width / w
                    new_width, new_height = width, int(h * scale)
                else:
                    scale = height / h
                    new_width, new_height = int(w * scale), height

                return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        

        def get_point(event, x, y, flags, param):
            if event == cv.EVENT_RBUTTONDOWN:
                if self.__line == 1:
                    self.__line = 2

                elif self.__line == 2:
                    self.__line = 1

            if event == cv.EVENT_LBUTTONDOWN:
                if self.__line == 1:
                    point_from_mouse[0] = (x, y)

                elif self.__line == 2:
                    point_from_mouse[1] = (x, y)

                image = cv.imread(self.__image_path)
        
        
        # read image, resize and error handling
        image = resize_with_aspect_ratio(self.__image, max_width)
        if image is None:
            print("Error: Image not found. Please check the path.")
            return

        original_image = image.copy()
        point_from_mouse = [None, None]

        cv.namedWindow("Image")
        cv.setMouseCallback("Image", get_point)


        while True:
            image = original_image.copy()

            if point_from_mouse[0] is not None:
                cv.line(
                    image,
                    tuple(map(int, point_from_mouse[0])),
                    tuple(map(int, vanishing_point)),
                    (0, 0, 255),
                    1,
                    lineType=cv.LINE_AA
                )

            if point_from_mouse[1] is not None:
                cv.line(
                    image,
                    tuple(map(int, point_from_mouse[1])),
                    tuple(map(int, vanishing_point)),
                    (0, 255, 0),
                    1,
                    lineType=cv.LINE_AA
                )

            cv.imshow("Image", image)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('s'):
                cv.imwrite('./final.jpg', image)

        cv.destroyAllWindows()


    def run_as(self, type):
        if type == "static":
            line1, line2 = self.static_line_detection()
        elif type == "auto":
            line1, line2 = self.auho_line_detection()

        try:
            vanp = self.find_vanishing_point(line1, line2)
            self.draw_line(vanp)
        except:
            print("ERROR: Wrong type")
