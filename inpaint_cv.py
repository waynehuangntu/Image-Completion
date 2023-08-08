import cv2
import config as cfg
def inpaint(image_File, image_Mask, radius = 3):
    image = cv2.imread(image_File)
    mask = cv2.imread(image_Mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    output = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete_cv.png", output)
    return output