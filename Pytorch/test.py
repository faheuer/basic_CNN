import numpy as np
from util import noisy_circle, find_circle, iou

def main():
    img_list = []
    param_list = []
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        #normalize
        img_list.append(img/img.max())
        param_list.append(params)
    param_list = np.array(param_list)
    img_list = np.array(img_list)

    # pass all samples as one batch for runtime
    detected = find_circle(img_list[:, :, :, np.newaxis])

    # z-transform prediction into data range
    detected[:,:2] = detected[:,:2]*200
    detected[:,2] = 10+detected[:,2]*40

    for i in range(1000):
        results.append(iou(param_list[i], detected[i]))
    results = np.array(results)
    print((results > 0.7).mean())

if __name__ == "__main__":
    main()
