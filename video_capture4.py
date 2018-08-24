# video_capture4.py
# !~/anaconda3/envs/fastai/bin/python python3
'''
Module that loads a pretrained resnet34 network, adds a custom head to predict 1 of 21 classes and a bounding box
then displays this in a video stream

Resnet is pretrained using the fastai deeplearning course pascal.ipynb notebook
get the weights stored in reg1.h5 from running that notebook

Most of this module runs pure pytorch, removing the fastai dependcy
'''
import cv2
from fastai.dataset import *
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

# We convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.
# do this after
def hw_bb(bb):
    return np.array([bb[1], bb[0], bb[3]+bb[1], bb[2]+bb[0]])
def bb_hw(a):
    return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])
def switchxy(a):
    '''
    numpy and pillow reverses x and y, this fixes it
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


def main():
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
    # create model, load weights from training in pascal_voc
    # choose a pretrained model
    mod = torchvision.models.resnet34(pretrained=True)
    # print(str(model))
    # a custom head for the model
    head_reg4 = nn.Sequential(
        Flatten(),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 4 + len(cats)),  # 4 numbers for the bounding box, plus one output per category
    )
    # strip off the av pool and FC layer
    mod = nn.Sequential(*list(mod.children())[:-2])
    # weld the custom head onto model (how to know its 8?)
    mod.add_module("8", head_reg4)
    # look at model architecture (the layer names are replaced with numbers?)
    print(str(mod))  # layer names stripped
    # load weights
    load_model(mod, '/home/keith/fastai2/fastai/data/VOCdevkit/VOC2007/models/reg1.h5')
    # set the model to eval which means we are predicting
    mod.eval()
    # transform the video frame appropriate to evaluation
    data_transforms = torchvision.transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # get the camera
    # (if cant get it it might be because its installed as root,first find it 'ls -la /dev/video*' see who owns it
    # if the account you are logged into does not own it then use sudo chown -R acct /dev/video0 to place video0
    # under acct's control)
    #
    cap = cv2.VideoCapture(0)
    # print FPS? If so uncomment this bit and the further bit in while loop below
    # print_fps = printFPS(cap)
    # img_no =0 #used to iterate through local test images
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # wait for a good frame
        if (ret == False):
            continue

        # frame = cv2.imread(str(img_no)+".jpg")
        # img_no= (img_no+1)%21

        img = frame

        # convert to PIL
        im = Image.fromarray(img)

        # run transforms on frame
        im_t = data_transforms(im)

        # convert to a batch size of 1
        im_t.unsqueeze_(0)

        # cv2.imshow('image', im_t)
        # convert to a variable
        img_variable = Variable(im_t)

        # forward pass
        pred = mod(img_variable)

        # normalize the coordinates
        # get size of mat
        height = frame.shape[0]
        width = frame.shape[1]

        # scale it to image width and height
        cords = expit((pred[0][:4]).detach())

        # output appears to be top left xy and bottom right xy
        cords2 = switchxy(cords)

        # makes it easier to see where points should be
        # drawGrid(frame, height, width)

        # scale bounding box by image size to get pixel values
        cords2 = convert_to_xy(cords2, width, height)

        # debug info
        print("coords:" + str(cords))  # show percentage of screen space
        print("coords2:" + str(cords2))  # show pixel location

        # get class
        cls_preds = (pred[0][4:]).detach().numpy()
        cls = np.argmax(cls_preds)  # list loc
        cls_val = cls_preds[cls]  # val

        # calculate the average of the predictions
        avg = np.average(cls_preds)
        std_dev = np.std(cls_preds)

        # only show prediction if really sure (not sure if this is valid)
        if (cls_val > (avg + 2.5 * std_dev)):
            # draw bounding box
            cv2.rectangle(frame, (int(cords2[0]), int(cords2[1])), (int(cords2[2]), int(cords2[3])), (255, 255, 255), 2)

            cv2.putText(frame, str(cats[cls + 1]),
                        (int(cords2[0]) + 5, int(cords2[1] + 25)),  # draw inside bounding box
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

if __name__ == '__main__':
    main()
