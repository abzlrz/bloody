import cv2, math
import numpy as np
from skimage import exposure, img_as_ubyte, feature
from sklearn import preprocessing
from PIL import Image, ImageEnhance

ycgcr = np.array([[  65.481,   128.553,  24.966  ],
                  [ -81.085,   112,     -30.915  ],
                  [  112,     -93.768,  -18.214  ]])


def rgb2ycgcr(rgb):
    rgb_arr = np.asarray(rgb, dtype=np.float)
    arr = np.dot(rgb_arr, ycgcr.T/255)

    arr[..., 0] += 16
    arr[..., 1] += 128
    arr[..., 2] += 128
    arr = np.asarray(arr, dtype='uint8')
    return arr

def feature_entropy(img):
    hist = cv2.calcHist(img, [0], None, [256], [0,256])
    hist_length = np.sum(hist)
    proba = [float(h) / hist_length for h in hist]
    
    entropy = -sum([P * math.log(P, 2) for P in proba if P != 0])
    return entropy

def feature_hist(image):
    hist = np.histogram(image)
    hist = np.hstack(hist).reshape(1, -1)
    hist = preprocessing.normalize(hist)
    features = np.hstack(hist)
    return features

def feature_colormoments(image):
    image = np.asarray(image, dtype='uint8')
    N = image.size
    mean = np.sum(1/N * image)
    standard_deviation = np.sqrt(1/N * np.sum((image - mean)**2))
    return mean, standard_deviation

def feature_hog(image):
    features = feature.hog(image,
                    orientations=9,
                    pixels_per_cell=(32,32),
                    cells_per_block=(1,1),
                    visualize=False,
                    block_norm='L2-Hys')
    return features

def preprocess(img):
    # CROPS IMAGE IN 1:1
    w, h = img.size
    if h > w:
        left = 0
        top = (h - w)/2
        right = w
        bottom = top + w
    else:
        left = (w - h)/2
        top = 0
        right = left + h 
        bottom = h
    img = img.crop((left, top, right, bottom))
    
    # RESIZE IMAGE TO 155X155
    img = img.resize((155, 155))
    
    # SATURATE IMAGE
    _filter = ImageEnhance.Color(img)
    img = _filter.enhance(2.0)

    # QUANTIZED IMAGE
    img = np.asarray(img)
    img = exposure.adjust_gamma(img)
    img = Image.fromarray(img)
    img = img.convert('P', palette=Image.ADAPTIVE, colors=64)
    img = img.convert('RGB')

    img = img_as_ubyte(img)
    return img

def extract_features(img):
    image = preprocess(img)
    ycc = rgb2ycgcr(image)

    __y = ycc[..., 0]
    _cg = ycc[..., 1]
    _cr = ycc[..., 2]

    # COLOR MOMENTS OF Y, CG, CR
    ym = feature_colormoments(__y)
    cgm = feature_colormoments(_cg)
    crm = feature_colormoments(_cr)
    cm = np.hstack([ym, cgm, crm])

    # HOG FEATURE OF CG, CR
    hog_cg = feature_hog(_cg)
    hog_cr = feature_hog(_cr)

    # COLOR HISTOGRAMS OF CG, CR
    ch_r = feature_hist(_cr)
    ch_g = feature_hist(_cg)

    # ENTROPY OF CR
    ent_cr = feature_entropy(_cr)
    features = np.hstack([cm, ch_r, ch_g, hog_cg, hog_cr, ent_cr])
    return features

import tqdm
def build_features(fnames, label=None):
    data = []
    for x in tqdm.tqdm(fnames):
        image = Image.open(x)
        try:
            features = extract_features(image)
            if label is None:
                data.append(features)
            else:
                data.append(np.hstack([features, label]))
        except ValueError:
            pass
    return data