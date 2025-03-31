import torch
from radstract.data.dicom import convert_dicom_to_images
from retuve.keyphrases.config import Config

from .utils import FILEDIR, generate_segmentation_objects, get_model, preprocess_images

MODEL_LOCATION_ULTRASOUND = f"{FILEDIR}/weights/hip-tinyunet-us.pth"

label_mapping = {
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
}


def get_tinyunet_model_us(config):
    model = get_model(MODEL_LOCATION_ULTRASOUND, config, no_classes=4)
    return model


def tinyunet_predict_dcm_us(dcm, keyphrase, model=None):
    config = Config.get_config(keyphrase)

    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )

    return tinyunet_predict_us(dicom_images, config, model)


def tinyunet_predict_us(imgs, keyphrase, model=None):
    return tinyunet_predict_us_shared(imgs, keyphrase, model, MODEL_LOCATION_ULTRASOUND)


def tinyunet_predict_us_shared(
    imgs, keyphrase, model=None, model_loc=None, batch_size=32
):
    config = Config.get_config(keyphrase)
    if model is None:
        model = get_model(model_loc, config)

    size = imgs[0].size
    print(f"Running inference on {len(imgs)} images")
    image_tensor = preprocess_images(imgs)
    with torch.inference_mode():
        image_tensor = image_tensor.to(config.device)

        # chunk the images into smaller batches (lets say )
        output = []
        for i in range(0, len(image_tensor), batch_size):
            output.append(model(image_tensor[i : i + batch_size]))

        output = torch.cat(output, dim=0)

        print(f"Ran inference on {len(imgs)} images")

        seg_results = []

        pred_masks = torch.argmax(output, dim=1)

        pred_masks = pred_masks.cpu().numpy()

        for i, pred_mask in enumerate(pred_masks):

            seg_frame_objects = generate_segmentation_objects(
                imgs[i].convert("RGB"),
                pred_mask,
                range(1, 4),
                (size[0], size[1]),
            )

            found_in_this_result = [seg_obj.cls for seg_obj in seg_frame_objects]

            seg_results.append(seg_frame_objects)

        return seg_results
