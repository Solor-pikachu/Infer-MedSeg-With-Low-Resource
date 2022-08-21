import glob
import SimpleITK as sitk
import numpy as np
import fastremap
from scipy.ndimage import binary_fill_holes

import cc3d 
import fastremap
def keep_largest_connected_object(class_probabilities,label):
    labels_in = class_probabilities==label
    labels_out = cc3d.connected_components(labels_in, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    unvalid = [i[0] for i in candidates[1:]]
    seg_map = np.in1d(labels_out, unvalid).reshape(labels_in.shape)
    return seg_map
def keep_largest_connected_area(class_probabilities,class_):
    seg_map = np.ones_like(class_probabilities)
    for i in class_:
        seg_map -= keep_largest_connected_object(class_probabilities,i)
    return class_probabilities*seg_map


def get_bbox_from_mask(mask, outside_value=0):
    mask_shape = mask.shape
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = max(int(np.min(mask_voxel_coords[0])) - int(20/space[2]),0)
    maxzidx = min(int(np.max(mask_voxel_coords[0])) + int(20/space[2]),mask_shape[0])
    
    minxidx = max(int(np.min(mask_voxel_coords[1])) - int(20/space[0]),0)
    maxxidx = min(int(np.max(mask_voxel_coords[1])) + int(20/space[0]),mask_shape[1])
    
    minyidx = max(int(np.min(mask_voxel_coords[2])) - int(20/space[1]),0)
    maxyidx = min(int(np.max(mask_voxel_coords[2])) + int(20/space[1]),mask_shape[2])
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]
def mask_to_bbox(image, mask,bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    image[resizer] = mask
    return image
paths = glob.glob('./nnUNet_raw_data/nnUNet_raw_data/Task666_FLARE/imagesTr/*')
for path in paths:
    
    img = sitk.ReadImage(path)
    
    
    origin = img.GetOrigin()
    space = img.GetSpacing()
    print(space)
    direction = img.GetDirection()
    
    img = sitk.GetArrayFromImage(img)
    seg = sitk.ReadImage(path.replace('imagesTr','labelsTr').replace('_0000.nii.gz','.nii.gz'))
    seg = sitk.GetArrayFromImage(seg)
    class_ = [ i in range(1,14)]
    seg = keep_largest_connected_area(seg, class_= class_) #保存最大联通区域
    
    mask = seg > 0
    bbox = get_bbox_from_mask(mask)
    
    img = crop_to_bbox(img, bbox)
    seg = crop_to_bbox(seg, bbox)
    
    seg_resized_itk = sitk.GetImageFromArray(img)
    seg_resized_itk.SetSpacing(space)
    seg_resized_itk.SetOrigin(origin)
    seg_resized_itk.SetDirection(direction)
    sitk.WriteImage(seg_resized_itk, path)
    
    seg_resized_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
    seg_resized_itk.SetSpacing(space)
    seg_resized_itk.SetOrigin(origin)
    seg_resized_itk.SetDirection(direction)
    sitk.WriteImage(seg_resized_itk, path.replace('imagesTr','labelsTr').replace('_0000.nii.gz','.nii.gz'))
    
#     break