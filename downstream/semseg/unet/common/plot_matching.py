import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch

def drawMatches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.

    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.

    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 1
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = (int(c[0]), int(c[1]), int(c[2]))
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img 

#def main(color_path, feature_path='/home/jhou/Dropbox/data', output_path='./output'):
#    plt.rcParams['figure.dpi'] = 300
#    os.makedirs(output_path, exist_ok=True)
#    for scene_id in os.listdir(feature_path):
#        os.makedirs(os.path.join(output_path, scene_id), exist_ok=True)
#        print(scene_id)
#        for image_pair in os.listdir(os.path.join(feature_path, scene_id)):
#            image_id1 = image_pair.split('_')[0]
#            image_id2 = image_pair.split('_')[1]
#            img1 = cv2.imread(os.path.join(color_path, '{}/color/{}.png'.format(scene_id, image_id1)))
#            img2 = cv2.imread(os.path.join(color_path, '{}/color/{}.png'.format(scene_id, image_id2)))
#            # resize image
#            img1 = cv2.resize(img1, (320,240), interpolation = cv2.INTER_AREA)
#            img2 = cv2.resize(img2, (320,240), interpolation = cv2.INTER_AREA)
#
#            features = torch.load(os.path.join(feature_path, scene_id, image_pair))
#            feature1 = features[0]
#            feature2 = features[1]
#            feature1 = feature1 / torch.norm(feature1, p=2, dim=0, keepdim=True)
#            feature2 = feature2 / torch.norm(feature2, p=2, dim=0, keepdim=True)
#
#            kp1 = []
#            kp2 = []
#            des1 = []
#            des2 = []
#            for x in range(feature1.shape[2]):
#                for y in range(feature1.shape[1]):
#                    kp1.append(cv2.KeyPoint(x*2,y*2,30))
#                    kp2.append(cv2.KeyPoint(x*2,y*2,30))
#                    des1.append(feature1[:,y,x].numpy())
#                    des2.append(feature2[:,y,x].numpy())
#            des1 = np.stack(des1)
#            des2 = np.stack(des2)
#
#            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#            matches = bf.match(des1,des2)
#            matches = sorted(matches, key = lambda x:x.distance)
#            #img3 = drawMatches(img1,kp1,img2,kp2,matches[:50], color=False)
#            img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
#            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#            plt.imsave(os.path.join(output_path, scene_id, '{}_{}.png'.format(image_id1, image_id2)), img3)

def main(color_path, feature_path='/home/jhou/Dropbox/data', output_path='./output'):
    plt.rcParams['figure.dpi'] = 300
    os.makedirs(output_path, exist_ok=True)
    for scene_id in os.listdir(feature_path):
        os.makedirs(os.path.join(output_path, scene_id), exist_ok=True)
        print(scene_id)
        for image_pair in os.listdir(os.path.join(feature_path, scene_id)):
            image_id1 = image_pair.split('_')[0]
            image_id2 = image_pair.split('_')[1]
            img1 = cv2.imread(os.path.join(color_path, '{}/color/{}.png'.format(scene_id, image_id1)))
            img2 = cv2.imread(os.path.join(color_path, '{}/color/{}.png'.format(scene_id, image_id2)))
            # resize image
            img1 = cv2.resize(img1, (320,240), interpolation = cv2.INTER_AREA)
            img2 = cv2.resize(img2, (320,240), interpolation = cv2.INTER_AREA)

            features = torch.load(os.path.join(feature_path, scene_id, image_pair))
            feature1 = features[0]
            feature2 = features[1]
            feature1 = feature1 / torch.norm(feature1, p=2, dim=0, keepdim=True)
            feature2 = feature2 / torch.norm(feature2, p=2, dim=0, keepdim=True)

            kp1 = []
            kp2 = []
            des1 = []
            des2 = []
            for x in range(feature1.shape[2]):
                for y in range(feature1.shape[1]):
                    kp1.append(cv2.KeyPoint(x*2,y*2,30))
                    kp2.append(cv2.KeyPoint(x*2,y*2,30))
                    des1.append(feature1[:,y,x].numpy())
                    des2.append(feature2[:,y,x].numpy())
            des1 = np.stack(des1)
            des2 = np.stack(des2)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            #img3 = drawMatches(img1,kp1,img2,kp2,matches[:50], color=False)
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            plt.imsave(os.path.join(output_path, scene_id, '{}_{}.png'.format(image_id1, image_id2)), img3)

if __name__ == "__main__":
    main(color_path='/checkpoint/jihou/data/scannet/partial_frames/', 
         feature_path='/checkpoint/jihou/clip3d/correspondence/chunk_features_resnet50/', 
         output_path='./output')
