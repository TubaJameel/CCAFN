import cv2
import numpy as np


class CLAHETransform:

    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):

        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, image):

        image = np.array(image)

        # Create CLAHE object HERE 
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.grid_size
        )

        # Convert RGB → Gray
        if len(image.shape) == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

        # Apply CLAHE
        image = clahe.apply(image)

        # Convert Gray → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        return image