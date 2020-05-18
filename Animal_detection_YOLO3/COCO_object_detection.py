from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _sigmoid(z):
    return 1/(1+np.exp(-z))

def load_img_pixel(file_name, shape):
    img = load_img(file_name)
    img_w, img_h = img.size
    img = load_img(file_name, target_size=(shape[0], shape[1]))
    img = img_to_array(img)
    img /= 255

    return img, img_w, img_h

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness, classes):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if(self.label == -1):
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if(self.score == -1):
            self.score = self.classes[self.label]
        return self.score

def decode_out(net_out, anchor, obj_thresh, net_w, net_h):
    grid_h, grid_w = net_out.shape[0], net_out.shape[1]
    nb_box = 3
    net_out = net_out.reshape(grid_h, grid_w, nb_box, -1)
    nb_class = net_out.shape[-1] - 5
    net_out[..., :2] = _sigmoid(net_out[..., :2])
    net_out[..., 4:] = _sigmoid(net_out[..., 4:])
    net_out[..., 5:] = net_out[..., 4][..., np.newaxis] * net_out[..., 5:]
    net_out[..., 5:] *= net_out[..., 5:] > obj_thresh
    boxes = []
    for i in range(grid_w*grid_h):
        row = int(i/grid_w)
        col = i%grid_w
        for b in range(nb_box):
            x, y, w, h = net_out[row][col][b][:4]
            objness = net_out[row][col][b][4]
            if(objness < 0.9 ):
                continue
            x = (x+col)/grid_w
            y = (y+row)/grid_h
            w = anchor[2*b+0]*np.exp(w)/net_w
            h = anchor[2*b+1]*np.exp(h)/net_h
            classes = net_out[row][col][b][5:]
            boxes.append( BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objness, classes))
    return boxes


def correct_box(boxes, net_w, net_h, img_w, img_h):
    if(float(net_w)/img_w) < (float(net_h)/img_h):
        new_w = net_w
        new_h = (net_w * img_h) / img_w

    else:
        new_h = net_h
        new_w = (net_h * img_w) / img_h

    for box in boxes:
        x_offset, x_scale = (net_w-new_w) / 2. / net_w , float(net_w)/new_w
        y_offset, y_scale = (net_h-new_h) / 2. / net_h , float(net_h)/new_h
        box.xmin = (box.xmin - x_offset) * x_scale * img_w
        box.xmax = (box.xmax - x_offset) * x_scale * img_w
        box.ymin = (box.ymin - y_offset) * y_scale * img_h
        box.ymax = (box.ymax - y_offset) * y_scale * img_h

def iou(box_1, box_2):
    x_min = max(box_1.xmin, box_2.xmin)
    y_min = max(box_1.ymin, box_2.ymin)
    x_max = min(box_1.xmax, box_2.xmax)
    y_max = min(box_1.ymax, box_2.ymax)
    inter = (float(x_max-x_min))*(y_max-y_min)
    union = (float(box_1.xmax - box_1.xmin)) * ( box_1.ymax - box_1.ymin) + (float(box_2.xmax - box_2.xmin)) * (box_2.ymax - box_2.ymin) - inter
    return float(inter)/union

def do_nms(boxes, threshold):
   m=0
   if(len(boxes)>0):
       nb_class = len(boxes[0].classes)
   else:
       return
   for c in range(nb_class):
       sorted_indice = np.argsort([-box.classes[c] for box in boxes])
       for i in range(len(sorted_indice)):
           if boxes[sorted_indice[i]].classes[c] == 0:     continue
           for j in range(i + 1, len(sorted_indice)):
               i_o_u = iou(boxes[sorted_indice[i]], boxes[sorted_indice[j]])
               if ( i_o_u >= threshold):
                   boxes[sorted_indice[j]].classes[c] = 0
               print(m)
               m+=1

def get_boxes(boxes, thresh, labels):
    v_boxes, v_scores, v_labels = list(), list(), list()
    for box in boxes:

        for i in range(len(box.classes)):
            if(box.classes[i] >= thresh):
                v_boxes.append(box)
                v_scores.append(box.classes[i] * 100)
                v_labels.append(labels[i])
    return v_boxes, v_scores, v_labels



def draw_boxes(filename, boxes, scores, labels):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xmin, box.ymin, box.xmax, box.ymax
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        ax.add_patch(rect)
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        print(label)
        plt.text(x1, y1, label, color='white')
    plt.show()

model = load_model('model.h5')
photo_file = 'fork.jpg'
input_w, input_h = 416, 416

img, w, h = load_img_pixel(photo_file, (416, 416))
img = img[None,...]
yhat = model.predict(img)
boxes = []

anchors = np.array([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]])

for i in range(len(yhat)):
    boxes += decode_out(yhat[i][0], anchors[i] , 0.6, input_w, input_h)

correct_box(boxes, input_w, input_h, w, h)
do_nms(boxes, 0.5)
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

v_boxes, v_scores, v_labels = get_boxes(boxes, 0.6, labels)
draw_boxes(photo_file, v_boxes, v_scores, v_labels)