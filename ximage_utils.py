import numpy as np
from ximage import ximread

def ximage_export(fpath, return_img=False, mode="single"):
    """
    Ximage export functionality
    convert ximage segmentations to indexmasks

    Parameters
    ----------
    fpath : str
        input file path.
    mode : str
        "single" or "multi"
        "single" mode returns a single image with different pixel values for each class
        "multi" mode returns multiple segmentation masks, one for each class
    return_img : bool
        if true returns (img, mask)
    Returns
    -------
    img (opt) : ndarray
        original img
    mask : ndarray
        segmentation mask

    """
    MODES = ["single", "multi"]
    if mode not in MODES:
        raise NotImplementedError(mode)
    
    im, im_meta = ximread(str(fpath))
    # create a lookup dictionary
    # {seg_class_name : (seg_class_index, seg_class_remapped_index)}
    classes_dict = {seg_class.name: (i, (i if seg_class.remap is None else seg_class.remap)) for i, seg_class in enumerate(im_meta.classes)}
    
    # create a reverse lookup dictionary
    # {class_index : class_remap_index}
    colormap = {seg_class_idx: (seg_class_remap,)for seg_class_idx, seg_class_remap in classes_dict.values()}
    
    if mode == "single":
        mask = np.full(im.shape[:2], 255, dtype=np.uint8)
        for item in im_meta.items:
            for blob in item.blobs:
                blob.draw(mask, colormap, True)
    elif mode == "multi":  
        # collect all blobs
        blobs = [blob for item in im_meta.items for blob in item.blobs]
        mask = {} 
        for seg_class, seg_class_idx in classes_dict.items():
            # NOTE: I'm indexing on remap, not sure if that's the correct thing to do
            blobs_per_class = [blob for blob in blobs if blob.get_classid() == seg_class_idx[1]]
            class_mask = np.full(im.shape[:2], 255, dtype=np.uint8)
            
            for blob in blobs_per_class:
                blob.draw(im=class_mask,
                          colormap={seg_class_idx[1] : (0,)},
                          filled=True)
            mask[seg_class] = class_mask
    
    if not return_img:
        return mask
    else:
        return im, mask
    