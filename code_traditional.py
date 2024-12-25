import cv2
import numpy as np
import math
import os
from PIL import Image
from tqdm import tqdm

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

def compute_disparity_map(left_image, right_image):
    # Window-based matching algorithm (block matching)
    window_size = 8 # Smaller for faster
    max_disparity = 256 # Smaller for faster
    h, w = left_image.shape
    disparity_map = np.zeros_like(left_image).astype(np.float32)

    for y in tqdm(range(window_size // 2, h - window_size // 2), desc='Processing', ncols=80):
        for x in range(window_size // 2, w - window_size // 2):
            best_diff = float('inf')
            best_disp = 0
            for d in range(max_disparity):
                if x - d - window_size // 2 < 0:
                    continue
                left_block = left_image[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1]
                right_block = right_image[y - window_size // 2:y + window_size // 2 + 1, x - d - window_size // 2:x + window_size // 2 + 1 - d]
                diff = np.sum(np.abs(left_block - right_block))
                if diff < best_diff:
                    best_diff = diff
                    best_disp = d
            disparity_map[y, x] = best_disp

    return disparity_map

def compute_disparity_map_opencv(left_image, right_image):
    # Sterero semi-global block matching algorithm in OpenCV
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=256,
                                   blockSize=8,
                                   P1=8 * 3 * 3 ** 2,
                                   P2=32 * 3 * 3 ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=15,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16
    return disparity_map

def test():
    test_imgs = ["Art", "Dolls", "Reindeer"]

    # Stereo image pair is already rectified?
    already_rectified = True

    # OpenCV Stereo Semi-Global Block Matching?
    opencv_method = True
    
    for index in range(3):

        left_view = "./PSNR_Assignment2/StereoMatchingTestings/" + test_imgs[index] + "/view1.png"
        right_view = "./PSNR_Assignment2/StereoMatchingTestings/" + test_imgs[index] + "/view5.png"

        gt_names = "./PSNR_Assignment2/PSNR_Python/gt/" + test_imgs[index] + "/disp1.png"  
        pred_names =  "./PSNR_Assignment2/PSNR_Python/pred/" + test_imgs[index] + f"/pred_disp1_{test_imgs[index]}.png"

        # Read the stereo image pair
        left_image, right_image = read_images(left_view, right_view)

        if not already_rectified:
            # Rectify the stereo image pair
            left_image, right_image = rectify_images(left_image, right_image)

        # Compute the disparity map
        if not opencv_method:
            pred_img = compute_disparity_map(left_image, right_image)
        else:
            pred_img = compute_disparity_map_opencv(left_image, right_image)

        # Save the disparity map
        cv2.imwrite(pred_names, pred_img)

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
        print(f'Image pair {test_imgs[index]}: the Peak-SNR value is {peaksnr}.\n')

if __name__ == "__main__":
    test()
