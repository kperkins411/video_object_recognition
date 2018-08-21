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
topLeftCornerOfText = (20,20)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

# We convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.
def hw_bb(bb):
    return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])
def bb_hw(a):
    return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

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

def denormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    # if not _is_tensor_image(tensor):
    #     raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(s)
    return tensor
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

#look at model architecture (the layer names are replaced with numbers
print(str(mod))   # layer names stripped

# import fastai.model as famd
# famd.model_summary(mod, (224,224))


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

#print FPS
print_fps = printFPS(cap)

def bb_hw(a):
    return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])
def hw_bb(bb):
    return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for a good frame
    if (ret == False):
        continue

    # You may need to convert the color.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to PIL
    im = Image.fromarray(img)

    # run transforms on frame
    im_t = data_transforms(im)

    #convert to a batch size of 1
    im_t.unsqueeze_(0)

    #convert to a variable
    img_variable = Variable(im_t)

    #forward pass
    pred = mod(img_variable)

    #normalize the coordinates
    # get size of mat
    height = frame.shape[0]
    width = frame.shape[1]

    cords = expit((pred[0][:4]).detach())

    cords2 = bb_hw(cords) #height right and left side right
    cords3 = (hw_bb(cords)) #very wrong
    print("coordinates "+ str(cords2))


    # draw bounding box
    cv2.rectangle(frame, (int(cords2[0]*width), int(cords2[1]*height)), (int(cords2[2]*width), int(cords2[3]*height)), (255, 0, 0), 2)

    #annotate class
    c = np.argmax((pred[0][4:]).detach().numpy())

    cv2.putText(frame, str(cats[c+1]),
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print_fps()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

