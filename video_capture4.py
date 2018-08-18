import cv2

class printFPS:
    def __init__(self, cap):
        self.modulo = 10    #print every 10th frame
        self.i = 0
        self.cap = cap

    def __call__(self):
        if(self.i%self.modulo==0):
            self.i=0
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        self.i+=1

cap = cv2.VideoCapture(1)
print_fps = printFPS(cap)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #find objects of interest in frame

    # upsize and display frame


    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # Display the resulting frame
    # cv2.imshow('frame',gray)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print_fps()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

