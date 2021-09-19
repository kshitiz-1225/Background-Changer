import torch
import numpy as np
from utility import inference
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from models.hrnet import hrnet
import argparse


# python main.py --image demo_images/image2.jpg --bg_image demo_images/background.jpg --weights hrnetv2_hrnet18_person_dataset_120.pth
# python main.py --image demo_images/image1.jpg --bg_image demo_images/background1.jpg --weights hrnetv2_hrnet18_person_dataset_120.pth


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        default="./",
        help="path to foreground image(single person image)",
    )

    parser.add_argument(
        "--bg_image",
        type=str,
        required=True,
        default="./",
        help="path to background image",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to save output image",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=" ",
    )

    args = parser.parse_args()

    person = Image.open(args.image)  # reads person image
    person = np.array(person)
    image = person.copy()

    bg = Image.open(args.bg_image)  # reads background image
    bg = np.array(bg)[:, :, :3]
    h, w, _ = image.shape
    hh, ww, cc = bg.shape

    # comparison between size
    if ww < w and hh < h:
        # if size of background < size of person image
        bg = cv2.resize(bg, (w, h))
        # bg is resized to person image
    elif ww < w:
        # if only width of background < width of person image
        bg = cv2.resize(bg, (w, hh))
        # bg width is resized to person image width
    elif hh < h:
        # if only height of background < height of person image
        bg = cv2.resize(bg, (ww, h))
        # bg height is resized to person image height

    #model is created
    model = hrnet(2)
    model.load_state_dict(
        torch.load(args.weights, map_location=torch.device("cpu"))[
            "state_dict"]
    )  # model is loaded
    model.eval()

    # for debugging shows images
    if args.debug:
        plt.imshow(image)
        plt.show()
        plt.imshow(bg)
        plt.show()

    with torch.no_grad():  # gradients are off
        # padding of image of person , so that both background image and person image are same
        xx = ww - w
        yy = hh - h
        xx = int((abs(xx)+xx)/2)
        yy = int((abs(yy)+yy)/2)
        ori_image = np.pad(image, ((yy//2, yy - yy//2), (xx//2, xx - xx//2), (0, 0)), 'constant',
                           constant_values=0)

        # output from semantic segmentation model
        prediction = inference(person, model)
        # from numpy convert to PIL format
        prediction = Image.fromarray(prediction)

        if args.debug:
            plt.imshow(prediction)
            plt.show()
        # prediction is converted to 3D array
        seg = np.zeros_like(image)
        seg[:, :, 0] = prediction

        seg[:, :, 1] = prediction
        seg[:, :, 2] = prediction

        seg = np.pad(seg, ((yy//2, yy - yy//2), (xx//2, xx - xx//2),  (0, 0)), 'constant',
                     constant_values=0)

        # array is made by keeping person image pixel where person is present else background image pixels
        result = np.where(seg, ori_image, bg)

        if args.debug:
            plt.imshow(result)
            plt.show()

        final_image = Image.fromarray(result, "RGB")

        final_image.save(f"{args.output_dir}/final.png")  # for Separate Use


if __name__ == "__main__":
    main()
