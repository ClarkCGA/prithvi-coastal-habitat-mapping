import numpy as np
import rasterio


def load_data(data_path, usage, is_label=False, apply_normalization=False, 
              global_stats=None, dtype=np.float32, verbose=False):
    r"""
    Open data using gdal, read it as an array and normalize it.

    Arguments:
            data_path (string): Full path including filename of the data source we wish to load.
            usage (string): Either "train", "validation", "inference".
            is_label (binary): If True then the layer is a ground truth (category index) and if
                                set to False the layer is a reflectance band.
            apply_normalization (binary): If true min/max normalization will be applied on each band.
            global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays 
                                 for each band. If not provided, these values will be calculated from the data.
            dtype (np.dtype): Data type of the output image chips.
            verbose (binary): if set to true, print a screen statement on the loaded band.

    Returns:
            image: Returns the loaded image as a 32-bit float numpy ndarray.
    """

    # Inform user of the file names being loaded from the Dataset.
    if verbose:
        print('loading file:{}'.format(data_path))

    # open dataset using rasterio library.
    with rasterio.open(data_path, "r") as src:

        if is_label:
            if src.count != 1:
                raise ValueError("Expected Label to have exactly one channel.")
            img = src.read(1)
            return img

        else:
            meta = src.meta
            if apply_normalization:
                img = do_normalization(src.read(), global_stats=global_stats)
                img = img.astype(dtype)
            else:
                img = src.read()
                img = img.astype(dtype)

    if usage in ["train", "validation"]:
        return img
    else:
        return img, meta


def do_normalization(img, global_stats):
    """
    Standardize the input image pixels.

    Args:
        img (np.ndarray): Stacked image bands with a dimension of (C, H, W).
        global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays
                      for each band. If not provided, these values will be calculated from the data.

    Returns:
        np.ndarray: Normalized image stack of size (C, H, W).
    """

    if global_stats is None:
        raise ValueError("Global statistics must be provided for global normalization.")
    else:
        means = np.array(global_stats['mean'])
        stds = np.array(global_stats['std'])
    
    normal_img = (img - means[:, None, None]) / stds[:, None, None]

    return normal_img


def flip(img, label, flip_type):
    r"""
    Synthesize a new pair of image, label chips by flipping the input chips around a user defined axis.

    Arguments:
            img (ndarray) -- Concatenated variables or brightness value with a dimension of (H,W,C)
            label (ndarray) -- Ground truth with a dimension of (H,W)
            flip_type (list) -- A flip type based on the choice of axis.
                                Provided transformation are:
                                    1) 'v_flip', vertical flip
                                    2) 'h_flip', horizontal flip
                                    3) 'd_flip', diagonal flip
    Returns:
            img -- A numpy array of flipped variables or brightness value.
            label --A numpy array of flipped labeled reference (ground truth).
    """

    def diagonal_flip(image):
        flipped = np.flip(image, 1)
        flipped = np.flip(flipped, 0)
        return flipped

    if isinstance(flip_type, str):
        # Horizontal flip
        if flip_type == "h_flip":
            img = np.flip(img, 0)
            label = np.flip(label, 0)

        # Vertical flip
        elif flip_type == "v_flip":
            img = np.flip(img, 1)
            label = np.flip(label, 1)

        # Diagonal flip
        elif flip_type == "d_flip":
            img = diagonal_flip(img)
            label = diagonal_flip(label)

        else:
            raise ValueError("Flip type must be one of: 'h_flip', 'v_flip' or 'd_flip'.")
    else:
        raise ValueError("Flip type param must be a tuple or list.")

    return img.copy(), label.copy()