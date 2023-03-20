import cv2
import numpy as np
import torch

class BilateralFilter(object):
    """Bilateral filter based on the OpenCV implementation.

    Args:
        d: Diameter of each pixel neighborhood that is used during filtering.
        sigmaColor: Filter sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
        sigmaSpace: Value of sigma in the coordinate space. The greater its value, the more further pixels will mix together, given 
                    that their colors lie within the sigmaColor range.
    """

    def __init__(self, d, sigmaColor, sigmaSpace):
        # Assert that the three input arguments are integers
        assert isinstance(d, int), "d must be an integer"
        assert isinstance(sigmaColor, int), "sigmaColor must be an integer"
        assert isinstance(sigmaSpace, int), "sigmaSpace must be an integer"
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = image.squeeze().numpy()
        # Divide the sigmaColor by 255 to correct for using float32 instead of uint8, see 
        # https://stackoverflow.com/questions/67023103/why-does-cv2-bilateralfilter-behave-so-differently-for-different-data-types
        filtered_img = cv2.bilateralFilter(image, self.d, self.sigmaColor/255, self.sigmaSpace)
        return torch.Tensor(filtered_img).unsqueeze(0)
