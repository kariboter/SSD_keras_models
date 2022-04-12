import cv2
from imutils.video import VideoStream


class Video:

    def __init__(self, url):
        '''
        here we initialize the class:
        width and height are dimensions of the stream
        half_width an arguments for slicing
        '''
        self.vs = VideoStream(src=url).start()
        self.width, self.height = (640, 240)
        self.half_width = 320
        self.grabbed = 0
        self.img_r = []
        self.img_l = []

    def get_splintered_video(self):
        frame = self.vs.read()
        self.img_r = frame[0:self.height, 0:self.half_width]
        self.img_l = frame[0:self.height, self.half_width:self.width]
        self.grabbed = self.vs.grabbed
        return self.img_r, self.img_l, self.grabbed

    def show_video(self):
        while True:
            self.get_splintered_video()
            cv2.imshow("Frame_r", self.img_r)
            cv2.imshow("Frame_l", self.img_l)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.stop()

