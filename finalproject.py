import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn
import os

images = []
num_images = 26
for i in range(num_images):
    images.append(cv2.imread(f'attempt4/capture{i+1}.JPG'))


# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def custom_zoooom(img, t, p1, p2, p3, p4):
    """
    Simple image zooming without boundary checking.
    Centered at "coord", if given, else the image center.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    coord: (float, float)
    """
    prev_x1, prev_y1 = p1
    prev_x2, prev_y2 = p2
    prev_x3, prev_y3 = p3
    prev_x4, prev_y4 = p4

    x1 = t * prev_x1
    x2 = len(img[0]) + t * (prev_x2 - len(img[0]))
    y1 = t * prev_y1
    y3 = len(img) + t * (prev_y3 - len(img))

    # Points to interpolate
    xs, ys = np.meshgrid((np.arange(img.shape[1]) / (img.shape[1] / (x2 - x1))) + x1, (np.arange(img.shape[0]) / (img.shape[0] / (y3 - y1))) + y1)

    # Points that represent the current image
    cur_image_points = (np.arange(img.shape[0]), np.arange(img.shape[1]))

    blue_interp = interpn(cur_image_points, img[:, :, 0], np.dstack([ys, xs]))
    green_interp = interpn(cur_image_points, img[:, :, 1], np.dstack([ys, xs]))
    red_interp = interpn(cur_image_points, img[:, :, 2], np.dstack([ys, xs]))

    # cv2.imshow("Thingy", np.dstack([blue_interp, green_interp, red_interp]) / 255)
    # cv2.waitKey(0)
    return np.dstack([blue_interp, green_interp, red_interp])

def get_homography(image1, image2):
    sift = cv2.SIFT_create()

    # create a mask image filled with zeros, the size of original image
    mask = 255 + np.zeros(image1.shape[:2], dtype=np.uint8)
    # mask[1500:2500, 2500:3500] = 0
    # mask = cv2.rectangle(mask, (2500,1500), (3500,2500), (0), thickness = -1)

    temp_image1 = np.copy(image1)
    temp_image2 = np.copy(image2)
    # temp_image1[1500:2500, 2500:3500, :] = 0
    # temp_image2[1000:3000, 2000:4000, :] = 0
    temp_image1[:2000, :, :] = 0
    temp_image2[:2000, :, :] = 0


    kp1, des1 = sift.detectAndCompute(temp_image1,None)
    kp2, des2 = sift.detectAndCompute(temp_image2,None)


    # Following 10 lines are mostly from OpenCV website
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Ratio Test from Lowe's paper
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # M is the homography matrix
    return M

def get_image_from_t(image, image2, homography, t):
    h,w = image.shape[:2]
    corners = np.array([[1, 1, 1], [w, 1, 1], [1, h, 1], [w, h, 1]]).T
    homographyApplied = (np.linalg.inv(homography) @ corners)
    newCoords = (homographyApplied[0:2] / homographyApplied[2]).astype(np.int32)

    # X1 is left, X2 is right, Y1 is top, Y2 is bottom
    avgX1, avgX2 = ((newCoords[0, 0] + newCoords[0, 2]) / 2), ((newCoords[0, 1] + newCoords[0, 3]) / 2)
    avgY1, avgY2 = ((newCoords[1, 0] + newCoords[1, 1]) / 2), ((newCoords[1, 2] + newCoords[1, 3]) / 2)

    # Basically estiamte the width and height and create a box at the center, just setting height equal to two thirds of width
    boxWidth, boxHeight = (avgX2 - avgX1)//2, (avgY2 - avgY1)//2
    boxHeight = boxWidth // 1.5
    estimated_coords = np.array([[3000 - boxWidth, 2000 - boxHeight], [3000 + boxWidth, 2000 - boxHeight], [3000 - boxWidth, 2000 + boxHeight], [3000 + boxWidth, 2000 + boxHeight]])

    print(estimated_coords)
    another_image = custom_zoooom(image, t, estimated_coords[0], estimated_coords[1], estimated_coords[2], estimated_coords[3])

    return another_image


# Calculate Homography Matrices
homography_matrices = []
for i in range(num_images - 1):
    # Load Homography from image 1 to image 2
    if not os.path.exists(f"attempt4/homography_test{i+1}.npz"):
        M = get_homography(images[i], images[i+1])
        homography_matrices.append(M)
        np.savez(f"attempt4/homography_test{i+1}.npz", M=M)
    else:
        M = np.load(f"attempt4/homography_test{i+1}.npz")["M"]
        homography_matrices.append(M)




# Calculate Intermediate Images
video = []
current_images = -1
number_of_frames = 26 * 15
for t in range(number_of_frames):
    print("T: " + str(t))
    if t % 15 == 0:
        current_images += 1
    
    if current_images == 5:
        another_image = get_image_from_t(images[current_images], images[current_images + 1], homography_matrices[current_images-1], 4*(t-(15 * (t//15))) / 15)
    else:
        another_image = get_image_from_t(images[current_images], images[current_images + 1], homography_matrices[current_images], 4*(t-(15 * (t//15))) / 15)
    video.append(another_image.astype(np.uint8))
    cv2.imwrite(f"res/{t}.jpg", another_image.astype(np.uint8))


video = np.array(video)

# Save Video
result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, (6000, 4000))
for i in range(number_of_frames):
    result.write(video[i]) 

result.release()