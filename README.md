# Statistics of Patch Offsets for Image Completion
## Usage Instructions
- Python Version – Python3
- OpenCV is required
- PyMaxflow must be installed in the system (pip install PyMaxflow)
- For GUI Interface run ```python main.py <image file name> GUI```
```
    python main.py floor.jpg GUI
```
- GUI Interface Keys:
```
    ESC: Shut Down
    r: Redraw
    space: Inpaint
```
- For Non-GUI Interface run ```python main.py <image file name> <mask file name>```, masks are contained in ./images/source folder and the output will be in ./images/output folder
```
    python main.py floor.jpg floor.png
```
#
## Notice

- The process may take some time, be patient!!
- Sometimes the process may fail and run into infinite loop owing to GUI interface, just restart the process.
- We print out informations on the terminal to monitor the process.


#

### Resources
1. He, Kaiming, and Jian Sun.: Statistics of patch offsets for image completion. ECCV (2012) 16-29.
2. He, K., Sun, J.: Computing nearest-neighbor fields via propagation-assisted kdtrees. CVPR (2012)
3. Boykov, Y., Veksler, O., Zabih, R., Fast approximate energy minimization via graph cuts. TPAMI (2001) 1222 - 1239
4. https://github.com/npinto/opencv/blob/master/samples/python2/common.py
5. https://github.com/chwahaha/statistic-of-similar-patch-offset
6. https://github.com/Pranshu258/Image_Completion

