import cv2
import numpy as np
import math
import os
from PIL import Image
from tqdm import tqdm
import subprocess

def psnr(img1, img2):
    # Peak Signal-to-Noise Ratio (PSNR) calculation
    mse = np.mean(((img1 - img2)) ** 2)
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def read_images(left_path, right_path):
    left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return left_image, right_image

def rectify_images(left_image, right_image):
    # Feature extraction with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    left_keypoints, left_descriptors = sift.detectAndCompute(left_image, None)
    right_keypoints, right_descriptors = sift.detectAndCompute(right_image, None)

    # Feature matching with BFMatcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)

    # Match filtering using the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Find the fundamental matrix
    src_pts = np.float32([left_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([right_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    F, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

    # Rectify the images
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2), F, left_image.shape[::-1])
    left_image = cv2.warpPerspective(left_image, H1, left_image.shape[::-1])
    right_image = cv2.warpPerspective(right_image, H2, right_image.shape[::-1])

    return left_image, right_image

def compute_disparity_map_psmnet(left_image_path, right_image_path):
    # Compute the disparity map using PSMNet
    cmd = (f"python ./PSMNet-master/Test_img.py "
           f"--loadmodel ./PSMNet-master/pretrained_sceneflow_new.tar "
           f"--leftimg {left_image_path} "
           f"--rightimg {right_image_path}")
    subprocess.run(cmd, shell=True, check=True)
    disparity_map = cv2.imread("Test_disparity.png", cv2.IMREAD_GRAYSCALE)
    return disparity_map

def resize_image(image_path, scaling_factor=0.5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image
    image_height, image_width = image.shape
    new_image_width = int(image_width * scaling_factor)
    new_image_height = int(image_height * scaling_factor)
    image_resized = cv2.resize(image, (new_image_width, new_image_height), interpolation=cv2.INTER_AREA)

    # Save the resized image
    resized_image_path = image_path[:-4] + "_resized.png"
    cv2.imwrite(resized_image_path, image_resized)
    return resized_image_path

def test():
    test_imgs = ["Art", "Dolls", "Reindeer"]

    # Stereo image pair is already rectified?
    already_rectified = True

    for index in range(3):

        left_view = "./PSNR_Assignment2/StereoMatchingTestings/" + test_imgs[index] + "/view1.png"
        right_view = "./PSNR_Assignment2/StereoMatchingTestings/" + test_imgs[index] + "/view5.png"

        gt_names = "./PSNR_Assignment2/PSNR_Python/gt/" + test_imgs[index] + "/disp1.png"  
        pred_names =  "./PSNR_Assignment2/PSNR_Python/pred/" + test_imgs[index] + f"/pred_disp1_{test_imgs[index]}.png"

        # Resize the images
        scaling_factor = 0.7
        left_view = resize_image(left_view, scaling_factor=scaling_factor)
        right_view = resize_image(right_view, scaling_factor=scaling_factor)
        gt_names = resize_image(gt_names, scaling_factor=scaling_factor)

        # Read the stereo image pair
        left_image, right_image = read_images(left_view, right_view)

        if not already_rectified:
            # Rectify the stereo image pair
            left_image, right_image = rectify_images(left_image, right_image)

        # Compute the disparity map
        pred_img = compute_disparity_map_psmnet(left_view, right_view)

        # Save the disparity map
        if os.path.exists(pred_names):
            os.remove(pred_names)
        os.rename("./Test_disparity.png", pred_names)

        # Read the ground-truth disparity map
        gt_img = cv2.imread(gt_names, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Calculate PSNR against the ground-truth disparity map
        # When calculate the PSNR:
        # 1.) The pixels in ground-truth disparity map with '0' value will be neglected.
        # 2.) The left part region (1-250 columns) of view1 is not included as there is no
        #   corresponding pixels in the view5.
        [h,l] = gt_img.shape
        gt_img = gt_img[:, 250:l]
        pred_img = pred_img[:, 250:l]
        pred_img[gt_img==0] = 0

        peaksnr = psnr(gt_img, pred_img)
        print(f'Image pair {test_imgs[index]} (resized to {int(scaling_factor * 100)} percent): the Peak-SNR value is {peaksnr}.\n')

if __name__ == "__main__":
    test()
