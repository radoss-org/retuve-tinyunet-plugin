import numpy as np
import torch
from retuve.classes.seg import SegFrameObjects
from retuve.hip_xray.classes import LandmarksXRay
from retuve.keyphrases.config import Config

from .utils import (
    FILEDIR,
    fit_triangle_to_mask,
    generate_segmentation_objects,
    get_model,
    preprocess_images,
)

MODEL_LOCATION_XRAY = f"{FILEDIR}/weights/hip-tinyunet-xray.pth"

label_mapping = {
    (255, 0, 0): 1,
}


def get_tinyunet_model_xray(config):
    model = get_model(MODEL_LOCATION_XRAY, config, no_classes=2)
    return model


def tinyunet_predict_xray(imgs, keyphrase, model=None):
    config = Config.get_config(keyphrase)
    if model is None:
        model = get_model(MODEL_LOCATION_XRAY, config, no_classes=2)

    print(f"Running inference on {len(imgs)} images")
    image_tensor = preprocess_images(imgs)
    with torch.inference_mode():
        image_tensor = image_tensor.to(config.device)
        output = model(image_tensor)

        seg_results = []
        # convert to numpy
        output = torch.argmax(output, dim=1).cpu().numpy()
        for ret, img in zip(output, imgs):
            seg_frame_objects = SegFrameObjects(img=np.array(img))

            ret_2 = ret.copy()
            ret_2[:, ret_2.shape[1] // 2 :] = 0
            ret[:, : ret.shape[1] // 2] = 0

            for ret_x in [ret, ret_2]:

                seg_frame_objects_sub = generate_segmentation_objects(
                    img, ret_x, [1], (img.width, img.height)
                )

                seg_frame_objects.append(seg_frame_objects_sub[0])

            seg_results.append(seg_frame_objects)

        landmark_results = []
        for seg_frame_objects in seg_results:
            landmarks = LandmarksXRay()
            if len(seg_frame_objects) != 2:
                landmark_results.append(landmarks)
                continue

            tri_1, tri_2 = seg_frame_objects[0], seg_frame_objects[1]
            fem_l, pel_l_o, pel_l_i, fem_r, pel_r_o, pel_r_i = fit_triangle_to_mask(
                tri_1.points, tri_2.points
            )

            if fem_l is not None:
                landmarks.fem_l, landmarks.pel_l_o, landmarks.pel_l_i = (
                    pel_l_i,
                    pel_l_o,
                    pel_l_i,
                )
                landmarks.fem_r, landmarks.pel_r_o, landmarks.pel_r_i = (
                    pel_r_i,
                    pel_r_o,
                    pel_r_i,
                )

            landmark_results.append(landmarks)

    return landmark_results, seg_results
