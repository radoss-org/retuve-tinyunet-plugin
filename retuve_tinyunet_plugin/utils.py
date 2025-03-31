import os

# Set the environment variable to disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from radstract.data.dicom import convert_dicom_to_images
from radstract.datasets.polygon_utils import segmentation_to_polygons
from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.keyphrases.config import Config
from stem_cv.models.tiny_unet import TinyUNet

FILEDIR = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/")


# Function to initialize and save the predictor if not already present
def get_model(model_location, config=None, no_classes=4):

    if config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config.device

    model = TinyUNet(in_channels=1, num_classes=no_classes)
    model.load_state_dict(
        torch.load(model_location, map_location=device, weights_only=True)
    )
    # model = torch.jit.script(model)
    model.to(device)
    model.eval()
    return model


def preprocess_images(images, target_size=(256, 256)):
    """
    Preprocess the input image for inference.
    """
    # convert images to numpy array
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ],
    )

    images = [np.array(image) for image in images]
    transformed_images = []
    for image in images:
        # apply the transform to the images
        transformed = transform(image=image)
        # only keep first channel
        transformed["image"] = transformed["image"][0:1]
        transformed_images.append(transformed["image"])

    image_tensor = torch.stack(transformed_images)

    return image_tensor


def generate_segmentation_objects(img_original, ret, class_range, img_shape):
    """
    Generate segmentation objects from a prediction result.

    Parameters:
    - img_original: Original PIL.Image object.
    - ret: numpy array, the segmentation result.
    - class_range: Range of classes to process (e.g., range(1, 4)).
    - img_shape: Tuple (width, height) for resizing masks.

    Returns:
    - SegFrameObjects containing segmentation objects.
    """
    seg_frame_objects = SegFrameObjects(img=np.array(img_original))

    for clss in class_range:
        mask = (ret == clss).astype(np.uint8)  # Convert bools to 1's
        mask = cv2.resize(mask, img_shape, Image.NEAREST)

        mask_rgb_white = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255
        mask_rgb_red = mask_rgb_white.copy()  # Copy only if needed
        white_mask = (mask_rgb_white == [255, 255, 255]).all(axis=2)
        mask_rgb_red[white_mask] = [255, 0, 0]

        polygons = segmentation_to_polygons(mask_rgb_red)
        try:
            points = polygons[1][0]
            points_np = np.array(points, dtype=np.int32)
        except (IndexError, TypeError, KeyError):
            points = None  # or a default value

        box = None
        if points is not None and len(points) >= 3:
            x, y, w, h = cv2.boundingRect(points_np)
            box = np.array([x, y, x + w, y + h])

        if points is not None and len(points) >= 3:
            seg_obj = SegObject(points, clss - 1, mask_rgb_white, box=box)
            seg_obj.cls = HipLabelsUS(seg_obj.cls)
            seg_frame_objects.append(seg_obj)

    if len(seg_frame_objects) == 0:
        seg_frame_objects = SegFrameObjects.empty(img=np.array(img_original))

    return seg_frame_objects


def fit_triangle_to_mask(tri_1_points, tri_2_points):

    tri_1_points, _ = find_triangle_from_edges(tri_1_points)
    tri_2_points, _ = find_triangle_from_edges(tri_2_points)

    if tri_1_points is None or tri_2_points is None:
        return (None, None, None, None, None, None)

    most_left_point = 100000
    tri_left = None
    tri_right = None
    for point in tri_1_points + tri_2_points:
        if point[0] < most_left_point:
            most_left_point = point[0]
            tri_left = tri_1_points if point in tri_1_points else tri_2_points
            tri_right = tri_2_points if point in tri_1_points else tri_1_points

    # check that tri_left is the left triangle
    if tri_left[0][0] > tri_right[0][0]:
        tri_left, tri_right = tri_right, tri_left

    fem_l, pel_l_o, pel_l_i = define_points(tri_left)
    fem_r, pel_r_o, pel_r_i = define_points(tri_right)

    return fem_l, pel_l_o, pel_l_i, fem_r, pel_r_o, pel_r_i


def find_triangle_from_edges(points):
    # Means we are already passing in a processed triangle
    if len(points) == 3:
        triangle = np.array(points)
        return triangle, cv2.contourArea(triangle)

    contours = np.array([points], dtype=np.int32)

    if len(contours) == 0:
        return None, 0  # No contours found

    # Approximate the largest contour to a polygon
    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Check if the approximated contour has 3 vertices (triangle)
    if len(approx) == 3:
        triangle = np.array([approx[0][0], approx[1][0], approx[2][0]])
        return triangle, cv2.contourArea(triangle)
    else:
        return None, 0  # No triangle found


def define_points(triangle):
    # Convert all points to tuples
    triangle = [(int(point[0]), int(point[1])) for point in triangle]

    # Find the lowest point in the triangle
    lowest_point = max(triangle, key=lambda point: point[1])
    triangle.remove(lowest_point)

    # Find the leftmost point in the triangle
    highest_point = min(triangle, key=lambda point: point[1])
    triangle.remove(highest_point)

    # The last point is the one not picked
    remaining_point = triangle[0]

    return lowest_point, highest_point, remaining_point
