import cv2
from fastai.conv_learner import *
from fastai.dataset import *
import torchvision
from fastai.torch_imports import *
from torchvision import datasets, models, transforms
import utils

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
from scipy.special import expit

IMAGE_SIZE=224
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

class printFPS:
    '''
    used to calculate and print FPS
    '''
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

def drawGrid(frame, height, width, ticks=4):
    '''
    just drws a grid on the frame
    :param frame:
    :param height:
    :param width:
    :param ticks:
    :return:
    '''
    color = (155, 155, 155)
    ptsize = 1

    # first vertical
    inc = int(width / ticks)
    size = 0
    for i in range(ticks):
        cv2.line(frame, (size, 0), (size, height), color, ptsize)
        size += inc

    # then horizontal
    inc = int(height / ticks)
    size = 0
    for i in range(ticks):
        cv2.line(frame, (0, size), (width, size), color, ptsize)
        size += inc

# We convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.
# do this after
def hw_bb(bb):
    return np.array([bb[1], bb[0], bb[3]+bb[1], bb[2]+bb[0]])
def bb_hw(a):
    return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])
def switchxy(a):
    '''
    numpy and pillow revers x and y, this fixes it
    :param a:
    :return:
    '''
    return np.array([a[1],a[0],a[3],a[2]])
def reverse_x_dir(a, width):
    return np.array([width-a[0],a[1],width-a[2],a[3]])

def convert_to_xy(cords, width, height):
    '''
    scales relative coord values (between 0 and 1)
    by multiplying x*width and y*height
    :param cords:
    :return:
    '''
    for i in range(len(cords)):
        if i % 2 == 0:
            cords[i] *= width
        else:
            cords[i] *= height
    return cords




#######################################
# get a list of categories
import json

# read cats.json into cats dict
with open('cats.json', 'r') as f:
    cats = json.load(f)

# convert the string keys to int keys
def jsonKeys2str(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x

cats = jsonKeys2str(cats)

#######################################
#create model, load weights from training in pascal_voc

#choose a pretrained model
mod = torchvision.models.resnet34(pretrained=True)
# print(str(model))

# a custom head for the model
head_reg4 = nn.Sequential(
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(25088,256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256,4+len(cats)), # 4 numbers for the bounding box, plus one output per category
)

# strip off the av pool and FC layer
mod = nn.Sequential(*list(mod.children())[:-2])

#weld the custom head onto model (how to know its 8?)
mod.add_module("8",head_reg4)

#look at model architecture (the layer names are replaced with numbers?)
print(str(mod))   # layer names stripped

# load weights
load_model(mod,'/home/keith/fastai2/fastai/data/VOCdevkit/VOC2007/models/reg1.h5')

#set the model to eval which means we are predicting
mod.eval()

# transform the video frame appropriate to evaluation
data_transforms =  torchvision.transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# get the camera (use chown -R keith /dev/video0 if you cant get it
#often on linux its installed as root
cap = cv2.VideoCapture(0)

# print FPS? If so uncomment this bit and the further bit in while loop below
# print_fps = printFPS(cap)

# img_no =0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for a good frame
    if (ret == False):
        continue

    # You may need to convert the color.
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = cv2.imread(str(img_no)+".jpg")
    # img_no= (img_no+1)%21

    img = frame

    # convert to PIL
    im = Image.fromarray(img)

    # run transforms on frame
    im_t = data_transforms(im)

    #convert to a batch size of 1
    im_t.unsqueeze_(0)

    # cv2.imshow('image', im_t)
    #convert to a variable
    img_variable = Variable(im_t)

    #forward pass
    pred = mod(img_variable)

    #normalize the coordinates
    # get size of mat
    height = frame.shape[0]
    width = frame.shape[1]

    #scale it to image width and height
    cords = expit((pred[0][:4]).detach())

    #output appears to be top left xy and bottom right xy
    cords2 = switchxy(cords)

    #makes it easier to see where points should be
    # drawGrid(frame, height, width)

    #scale bounding box by image size to get pixel values
    cords2 =convert_to_xy(cords2, width, height)

    #debug info
    print("coords:" + str(cords))   #show percentage of screen space
    print("coords2:" + str(cords2)) #show pixel location

    #get class
    cls_preds = (pred[0][4:]).detach().numpy()
    cls = np.argmax(cls_preds)  #list loc
    cls_val = cls_preds[cls] #val

    #calculate the average of the predictions
    avg = np.average(cls_preds)
    std_dev = np.std(cls_preds)

    #only show prediction if really sure (not sure if this is valid)
    if (cls_val>(avg+2.5*std_dev)):
        # draw bounding box
        cv2.rectangle(frame, (int(cords2[0]), int(cords2[1])), (int(cords2[2]), int(cords2[3])), (255, 255, 255), 2)

        cv2.putText(frame, str(cats[cls+1]),
                    (int(cords2[0])+5, int(cords2[1]+25)),   #draw inside bounding box
                    font,
                    fontScale,
                    fontColor,
                    lineType)


    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print_fps()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

