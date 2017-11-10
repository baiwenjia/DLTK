import SimpleITK as sitk
import tensorflow as tf
import os

from dltk.io.augmentation import *
from dltk.io.preprocessing import *


def crop_image(image, cx, cy, size):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)), 'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)), 'constant')
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def receiver(file_references, mode, params=None):
    """Summary
    
    Args:
        file_references (TYPE): Description
        mode (TYPE): Description
        params (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    
    def _augment(img, lbl):
        
        img = add_gaussian_noise(img, sigma=0.1)
        [img, lbl] = flip([img, lbl], axis=1)
        
        return img, lbl

    n_examples = params['n_examples']
    example_size = params['example_size']

    i = 0
    while True:

        img_fn = file_references[i]
        i += 1
        if i == len(file_references):
            i = 0

        images = sitk.GetArrayFromImage(sitk.ReadImage(img_fn))

        # Normalise volume images
        images = normalise_zero_one(images)

        # Crop the image
        cx, cy = int(images.shape[0] / 2), int(images.shape[1] / 2)
        images = crop_image(images, cx, cy, example_size)

        # Add a channel dimension
        images = np.expand_dims(images, axis=-1)

        # Move the z dimension to the batch dimension
        images = np.transpose(image, (2, 0, 1, 3))

        # Add the z dimension as DLTK uses 3D network by default, which requires the shape [batch, x, y, z, channel]
        images = np.expand_dims(images, axis=2)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'labels': None}

        lbl = sitk.GetArrayFromImage(sitk.ReadImage(re.sub('sa', 'label_sa', img_fn)))
        lbl = crop_image(lbl, cx, cy, example_size)
        lbl = np.transpose(lbl, (2, 0, 1))
        lbl = np.expand_dims(lbl, axis=2)

        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, lbl = _augment(images, lbl)
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images, lbl = extract_class_balanced_example_array(images, lbl, example_size=example_size,
                                                               n_examples=n_examples, classes=4)
            for e in range(n_examples):
                yield {'features': {'x': images[e].astype(np.float32)}, 'labels': {'y': lbl[e].astype(np.int32)}}
        else:
            yield {'features': {'x': images}, 'labels': {'y': lbl}}

    return


def save_fn(file_reference, data, output_path):
    """Summary
    
    Args:
        file_references (TYPE): Description
        data (TYPE): Description
        output_path (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    lbl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(file_reference), 'LabelsForTraining.nii')))

    new_sitk = sitk.GetImageFromArray(data)

    new_sitk.CopyInformation(lbl)

    sitk.WriteImage(new_sitk, output_path)