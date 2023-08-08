"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import kdtree
import energy
import operator
import numpy as np
import config as cfg
import inpaint_cv as cv
import GUI
from time import time
from scipy import ndimage
from sklearn.decomposition import PCA

def GetBoundingBox(mask):
    """
    Get Bounding Box for a Binary Mask
    Arguments: mask - a binary mask
    Returns: col_min, col_max, row_min, row_max
    """
    start = time()
    #white region is the masked region
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if cfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255,255,255), 1)
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + cfg.BB_IMAGE_SUFFIX, mask)
    end = time()
    print ("GetBoundingBox execution time: ", end - start)
    return bbox

def GetSearchDomain(shape, bbox):
    """
    get a rectangle that is 3 times larger (in length) than the bounding box of the hole
    this is the region which will be used for the extracting the patches
    """
    start = time()
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], shape[0]-1)
    end = time()
    print ("GetSearchDomain execution time: ", end - start)
    return col_min, col_max, row_min, row_max

def GetPatches(image, bbox, hole):
    """
    args:
    bbox: seach domain, larger than bounding box
    hole: the real bounding box
    get the patches from the search region in the input image
    """
    start = time()
    indices, patches = [], []
    rows, cols, _ = image.shape
    for i in range(int(bbox[2])+cfg.PATCH_SIZE//2, int(bbox[3])-cfg.PATCH_SIZE//2):
        for j in range(int(bbox[0])+cfg.PATCH_SIZE//2, int(bbox[1])-cfg.PATCH_SIZE//2):
            if i not in range(int(hole[2])-cfg.PATCH_SIZE//2, int(hole[3])+cfg.PATCH_SIZE//2) and j not in range(int(hole[0])-cfg.PATCH_SIZE//2, int(hole[1])+cfg.PATCH_SIZE//2):
                indices.append([i,j])
                patches.append(image[i-cfg.PATCH_SIZE//2:i+cfg.PATCH_SIZE//2, j-cfg.PATCH_SIZE//2:j+cfg.PATCH_SIZE//2].flatten())
    end = time()
    print ("GetPatches execution time: ", end - start)
    return np.array(indices), np.array(patches, dtype='int64')

def ReduceDimension(patches):
    start = time()
    pca = PCA(n_components=24)
    reducedPatches = pca.fit_transform(patches)
    end = time()
    print ("ReduceDimension execution time: ", end - start)
    return reducedPatches

def GetOffsets(patches, indices):
    start = time()
    kd = kdtree.KDTree(patches, leafsize=cfg.KDT_LEAF_SIZE, tau=cfg.TAU)
    dist, offsets = kdtree.get_annf_offsets(patches, indices, kd.tree, cfg.TAU)
    end = time()
    print ("GetOffsets execution time: ", end - start)
    return offsets

def GetKDominantOffsets(offsets, K, height, width):
    """
    getting the peak offsets
    """
    start = time()
    x, y = [offset[0] for offset in offsets if offset != None], [offset[1] for offset in offsets if offset != None]
    #bins = [[i for i in range(np.min(x),np.max(x))], [i for i in range(np.min(y),np.max(y))]]
    #bins = []
    tmp1 = []
    tmp2 = []
    for i in range(np.min(x),np.max(x)):
        tmp1.append(i)
    for i in range(np.min(y),np.max(y)):
        tmp2.append(i)
    bins = [tmp1, tmp2]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = hist.T
    
    p, q = np.where(hist == cv2.dilate(hist, np.ones(8))) # Non Maximal Suppression
    nMSHist = np.zeros(hist.shape)
    nMSHist[p, q] = hist[p, q]
    
    p, q = np.where(nMSHist >= np.partition(nMSHist.flatten(), -K)[-K])
    peakHist = np.zeros(hist.shape)
    peakHist[p, q] = nMSHist[p, q]
    
    peakOffsets, freq = [[xedges[j], yedges[i]] for (i, j) in zip(p, q)], nMSHist[p, q].flatten()
    peakOffsets = np.array([x for _, x in sorted(zip(freq, peakOffsets), reverse=True)], dtype="int64")[:2*K]
    end = time()
   
    print ("GetKDominantOffsets execution time: ", end - start)
    return peakOffsets 

def GetOptimizedLabels(image, mask, labels):
    start = time()
    optimizer = energy.Optimizer(image, mask, labels)
    sites, optimalLabels = optimizer.Init_Labelling()
    optimalLabels = optimizer.OptimizeLabellingABS(optimalLabels)
    end = time()
    print ("GetOptimizedLabels execution time: ", end - start)
    return sites, optimalLabels 

def CompleteImage(image, sites, mask, offsets, optimalLabels):
    failedPoints = mask
    completedPoints = np.zeros(image.shape)
    finalImg = image
    for i in range(len(sites)):
        j = optimalLabels[i]
        finalImg[sites[i][0], sites[i][1]] = image[sites[i][0] + offsets[j][0], sites[i][1] + offsets[j][1]]
        completedPoints[sites[i][0], sites[i][1]] = finalImg[sites[i][0], sites[i][1]]
        failedPoints[sites[i][0], sites[i][1]] = 0
    return finalImg, failedPoints, completedPoints

def PoissonBlending(image, mask, center):
    #getting better results with poisson blending
    src = cv2.imread(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png")
    dst = cv2.imread(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png")
    #print(type(src))
    blendedImage = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
    return blendedImage


def main(imageFile, maskFile):
    """
    Image Completion Pipeline
        1. Patch Extraction
        2. Patch Offsets
        3. Image Stacking
        4. Blending
    """
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    imageR = cv2.imread(imageFile)
    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    bb = GetBoundingBox(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cfg.TAU = max(bbwidth, bbheight)//15
    cfg.DEFLAT_FACTOR = image.shape[1]
    search_domain = GetSearchDomain(image.shape, bb)

    indices, patches = GetPatches(imageR, search_domain, bb)
    #print(patches[0])
    reducedPatches = ReduceDimension(patches)
    #print(reducedPatches[0])
    offsets = GetOffsets(reducedPatches, indices)
    kDominantOffset = GetKDominantOffsets(offsets, 60, image.shape[0], image.shape[1])
    sites, optimalLabels = GetOptimizedLabels(imageR, mask, kDominantOffset)
    completedImage, failedPoints, completedPoints = CompleteImage(imageR, sites, mask, kDominantOffset, optimalLabels)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png", completedImage)
    print(f"Finish saving completion result.")
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png", completedPoints)
    center = (bb[2]+bbwidth//2, bb[0]+bbheight//2)
    blendedImage = PoissonBlending(imageR, mask,center)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_blendedImage.png", blendedImage)
    print(f"Finish saving blended result.")
    if (np.sum(failedPoints)):
        #print("Image Completion failed, please try it again, it may require multiple times.")
        #cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Failed.png", failedPoints)
        #main(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png", cfg.OUT_FOLDER + cfg.IMAGE + "_Failed.png")
        _ = cv.inpaint(imageFile, maskFile)
        #return cv.inpaint(imageFile, maskFile)
    

def GUI_main(imageFile, mask):
    """
    Image Completion Pipeline
        1. Patch Extraction
        2. Patch Offsets
        3. Image Stacking
        4. Blending
    """
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    imageR = cv2.imread(imageFile)
    #mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    bb = GetBoundingBox(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cfg.TAU = max(bbwidth, bbheight)//15
    cfg.DEFLAT_FACTOR = image.shape[1]
    search_domain = GetSearchDomain(image.shape, bb)

    indices, patches = GetPatches(imageR, search_domain, bb)
    #print(patches[0])
    reducedPatches = ReduceDimension(patches)
    #print(reducedPatches[0])
    offsets = GetOffsets(reducedPatches, indices)
    kDominantOffset = GetKDominantOffsets(offsets, 60, image.shape[0], image.shape[1])
    sites, optimalLabels = GetOptimizedLabels(imageR, mask, kDominantOffset)
    completedImage, failedPoints, completedPoints = CompleteImage(imageR, sites, mask, kDominantOffset, optimalLabels)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png", completedImage)
    print(f"Finish saving completion result.")
    #cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png", completedPoints)
    center = (bb[2]+bbwidth//2, bb[0]+bbheight//2)
    blendedImage = PoissonBlending(imageR, mask,center)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_blendedImage.png", blendedImage)
    print(f"Finish saving blended result.")
    if (np.sum(failedPoints)):
        return cv.inpaint(imageFile, maskFile)
    return completedImage
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ("Usage: python main.py image_name mask_file_name")
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    print (f"Image file = {imageFile}")
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    print(f"Mask file = {maskFile}")
    if len(sys.argv) == 4 and sys.argv[3] == "cv":
        print("using cv")
        ret = cv.inpaint(imageFile, maskFile)
    elif len(sys.argv) == 3 and sys.argv[2] == "GUI":
        print("GUI Interface")
        img = cv2.imread(imageFile)
        img_mark = img.copy()
        mark = np.zeros(img.shape[:2], np.uint8)
        sketch = GUI.Sketcher('img', [img_mark, mark], lambda : ((255, 255, 255), 255))

        while True:
            ch = 0xFF & cv2.waitKey()
            if ch == 27:
                break
            if ch == ord(' '):
                #res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
                res = GUI_main(imageFile, mark)
                cv2.imshow('inpaint', res)
            if ch == ord('r'):
                img_mark[:] = img
                mark[:] = 0
                sketch.show()
        cv2.destroyAllWindows() 
    else:
        main(imageFile, maskFile)
    