from typing import Union, Tuple, List
import numpy as np
from skimage import measure
import torch

def slice_argmax(class_probabilities,step=5):
    C,Z,X,Y = class_probabilities.shape
    result = np.zeros((Z,X,Y))
    z = class_probabilities.shape[1]
    stride = int(z/step)
    step1 = [i*stride for i in range(step)]+[z]
    for i in range(step):
        result[step1[i]:step1[i+1]] = torch.argmax(class_probabilities[:,step1[i]:step1[i+1]],0).cpu().numpy()
    torch.cuda.empty_cache()
    return result

def resize_and_argmax(class_probabilities,size_after_cropping,step=4): #切片resize
    result = np.zeros(size_after_cropping)
    z = class_probabilities.shape[1]
    stride = int(z/step)
    step1 = [i*stride for i in range(step)]+[z]
    z = size_after_cropping[0]
    stride = int(z/step)
    step2 = [i*stride for i in range(step)]+[z]

    for i in range(step):
        size = size_after_cropping
        size[0] = step2[i+1] - step2[i]
        slicer = class_probabilities[:,step1[i]:step1[i+1]][None].half()
        slicer = torch.nn.functional.interpolate(slicer.cuda(),mode='trilinear',size=size, align_corners=True)[0]
        result[step2[i]:step2[i+1]] = slice_argmax(slicer)
        del slicer
        torch.cuda.empty_cache()
    return result



def distance(a,b):
    a = a[0]
    b = b[0]
    return np.sqrt((abs(a-b)*abs(a-b)).sum())
def get_centroid(seg,center_class=5):
    binary = seg == center_class
    label_prob_ = measure.label(binary,connectivity=3)
    region_label_prob_ = measure.regionprops(label_prob_)
    if len(region_label_prob_) == 0:
        return None
    areas = []
    for region in region_label_prob_:
        areas.append(region.area)
    if len(areas) == 0:
        return None
    else:
        return region_label_prob_[np.argmax(areas)].centroid

def del_fp(seg,sample_centroid,score=70):
    binary = seg > 0
    label_prob_ = measure.label(binary,connectivity=3)
    region_label_prob_ = measure.regionprops(label_prob_)
    valid_label = []
    for region in region_label_prob_:
        dis_ = distance(np.array(region.centroid),np.array(sample_centroid))
        if dis_ > score:
            valid_label.append(region.label)
    if len(valid_label) > 0:
        seg_map = ~np.in1d(label_prob_, valid_label).reshape(label_prob_.shape)
        seg *= seg_map
    return seg
def keep_max_conpont(seg,class_=1):
    binary = seg ==  class_
    label_prob_ = measure.label(binary,connectivity=3)
    region_label_prob_ = measure.regionprops(label_prob_)
    if len(region_label_prob_) == 0 or len(region_label_prob_) == 1:
        return seg
    try:
        areas = []
        for region in region_label_prob_:
            areas.append(region.area)

        valid_label = []
        big = np.max(areas)
        for region in region_label_prob_:
            if region.area < big:
                valid_label.append(region.label)
        if len(valid_label) > 0:
            seg_map = ~np.in1d(label_prob_, valid_label).reshape(label_prob_.shape)
            seg *= seg_map
    except:
        pass
    return seg

def detect_tp(seg,classe):
    binary = seg == classe
    label_prob_ = measure.label(binary,connectivity=3)
    region_label_prob_ = measure.regionprops(label_prob_)
    if len(region_label_prob_) > 0:
        return True
    else:
        return False

def del_min_conpoent(seg_old_spacing,voxel=21):
    binary = seg_old_spacing>0
    label_prob = measure.label(binary,connectivity=3)
    region_label_prob = measure.regionprops(label_prob)
    valid_label = []
    for region in region_label_prob:
        if region.area < 21:
            valid_label.append(region.label)
            
    if len(valid_label) > 0:
        seg_map = ~np.in1d(label_prob, valid_label).reshape(label_prob.shape)
        seg_old_spacing*=seg_map
    return seg_old_spacing
def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
        res = torch.from_numpy(res)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer
    
def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps