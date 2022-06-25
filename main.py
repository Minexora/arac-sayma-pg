import cv2
import numpy as np
from time import sleep


class VehicleCounting:
    min_rec_width = 80  # Minimum dikdörtgen genişliği
    min_rec_height = 80  # Dikdörtgenin minimum yüksekliği
    offset = 6  # Hata payı(pixsel)
    detect = []  # Tespit edilen araç konumları (izlemek için)
    delay = 60  # Fps
    cars_counter = 0  # Araç sayısı
    line_top = 550  # Çizginin yukarıdan yüksekliği
    line_width = 600  # Çizgi genişliği
    line_left = 625  # Çizginin soldan kaç pixel olacağı

    def __init__(self, video_url='video.mp4') -> None:
        self.cap = cv2.VideoCapture(video_url)
        self.subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

        while True:
            ret, frame1 = self.cap.read()
            waiting_time = float(1 / self.delay)
            sleep(waiting_time)
            grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey, (3, 3), 5)
            img_sub = self.subtractor.apply(blur)
            expansion = cv2.dilate(img_sub, np.ones((5, 5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morphological_transformations = cv2.morphologyEx(expansion, cv2. MORPH_CLOSE, kernel)
            morphological_transformations = cv2.morphologyEx(morphological_transformations, cv2. MORPH_CLOSE, kernel)
            contour, h = cv2.findContours(morphological_transformations, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            self.define_line(frame=frame1)
            self.examination_contours(contour, h, frame1)

            self.show_frame(frame1, morphological_transformations)

            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def define_line(self, frame, color=False):
        if(not color):
            cv2.line(frame, (self.line_left + self.line_width, self.line_top), (self.line_width, self.line_top), (255, 127, 0), 3)
        else:
            cv2.line(frame, (self.line_left * self.line_width, self.line_top), (self.line_width, self.line_top), (0, 127, 255), 3)

    def examination_contours(self, contour, h, frame1):
        for(i, c) in enumerate(contour):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_contour = (w >= self.min_rec_width) and (h >= self.min_rec_height)
            if not validate_contour:
                continue

            # cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            center = self.find_center(x, y, w, h)
            if (self.line_left <= center[0]):
                self.detect.append(center)
            # cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            for (x, y) in self.detect:
                if y < (self.line_top + self.offset) and y > (self.line_top - self.offset):
                    self.cars_counter += 1
                    self.define_line(frame=frame1, color=True)
                    self.detect.remove((x, y))
                    print("car is detected : "+str(self.cars_counter))

    def find_center(self, x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    def show_frame(self, frame1, morphological_transformations):
        cv2.putText(frame1, "VEHICLE COUNT : "+str(self.cars_counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow("Video Original", frame1)
        cv2.imshow("Detectar", morphological_transformations)


VehicleCounting()
