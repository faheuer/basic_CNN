
import numpy as np
from shapely.geometry import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

import torchvision.models as models
import torch.nn as nn
import torch

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img_list):
    # predict circle params with network
    model = create_net()
    model.load_weights("model.h5")
    prediction = model.predict(img_list)
    return prediction


def euclidean_distance(params0, params1):
    # metric 1
    euclidean = K.sqrt(K.sum(K.square(params0[:2]*200 - params1[:2]*200), axis=-1))
    return euclidean

def circle_size_diff(params0, params1):
    # metric 2
    size_difference = K.abs(params0[3] - params1[3])
    # range: [10~50]
    return size_difference * 40


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )

def generate_batch(batch_size, noise_level):
    while True:
        # initiate empty for runtime
        image_batch = np.zeros((batch_size, 3, 200, 200))
        label_batch = np.zeros((batch_size, 3))
        for i in range(batch_size):
            params, img = noisy_circle(200, 50, noise_level)

            # normalize data
            image_batch[i] = np.array([img, img, img])/img.max()
            label_batch[i] = [params[0]/200, params[1]/200, (params[2]-10)/40]

        yield (torch.from_numpy(image_batch), torch.from_numpy(label_batch))


def create_net():

    model = models.resnet18(pretrained=True)
    #set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    input_size = 200
    return model#, input_size
