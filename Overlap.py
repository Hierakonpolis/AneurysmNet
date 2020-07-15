import numpy as np
import nibabel as nib
import math
from pathlib import Path
import pickle

def niload(path):
    """
    Returns a numpy object from the nifti file path
 
    Parameters
    ----------
    path : str
        Path to the nifty file.
 
    Returns
    -------
    new : numpy.ndarray
        Nifti volume as a numpy object.
 
    """
    nob=nib.load(path, keep_file_open=False)
    data=nob.get_fdata()
    return data

def GetIndexes(volume,loc,size):
   
    max_upper_indexes=np.array(volume.shape)-size
   
    lows=np.clip(loc-np.array(size/2).astype(int),0,max_upper_indexes).astype(int)
   
    highs=lows+size
   
    return lows, highs, loc

def Carve(low,high,vol):
   
    return vol[low[0]:high[0],low[1]:high[1],low[2]:high[2]]

def overlapping_patches(patches, box_size, coordinates):
    """
    Finds overlapping patches and counts the average for overlapping voxels

    Parameters
    ----------
    patches : ndarray
        An array containing the patches proposed by the neural network.

    box_size : int
        Length of one side of the box around the center coordinates.

    coordinates : ndarray
        An array containing the coordinates for the center points of the patches in respective order to given patches

    Returns
    -------
    new : ndarray
        The original patches-ndarray with the overlapping voxel-values changed to average.

    """
    averaged = ([])
    coordinates = coordinates.T

    while coordinates.size > 3:
        first_co, coordinates = np.split(coordinates, [1], 1)

        indexes_x = np.nonzero(abs(coordinates[:1] - first_co[0]) <= box_size)
        indexes_y = np.nonzero(abs(coordinates[1:2] - first_co[1]) <= box_size)
        indexes_z = np.nonzero(abs(coordinates[2:3] - first_co[2]) <= box_size)

        indexes = indexes_x[1][np.isin(indexes_x[1], indexes_y[1])]
        indexes = indexes[np.isin(indexes, indexes_z[1])]

        if indexes.size > 0:
            add, patches = voxel_average(patches, indexes, coordinates, first_co, box_size)
        else:
            add = patches.pop(0)
        averaged.append(add)

    add = patches.pop(0)
    averaged.append(add)

    return averaged

def voxel_average(patches, indexes, coordinates, first_co, box_size):
    """
    Replaces the voxel values in overlapping regions with the average of overlapping voxels values.

    Parameters
    ----------
    patches : list
        An array containing the patches proposed by the neural network.

    indexes : ndarray
        An array containing the indexes of patches that overlap with the patch in index 0

    coordinates : ndarray
        An array containing the coordinates for the center points of the patches in respective order to given patches

    box_size : int
        Length of one side of the box around the center coordinates.

    Returns
    -------
    new : ndarray
        The patch in index 0 with the overlapping values changed.

    new : list
        A list containing the rest of the patches with the voxels overlapping with the first one changed.
    """
    first = patches.pop(0)
    first_co = np.array([first_co[0][0], first_co[1][0], first_co[2][0]])
    overlap_mask = np.ones((64, 64, 64))
    borders_list = []

    for index in indexes:
        overlap_borders = overlap_coord(coordinates[:, index], first_co, box_size)

        first_borders = overlap_box(first_co.T, overlap_borders, box_size)
        second_borders = overlap_box(coordinates[:, index], overlap_borders, box_size)

        overlap_mask = add_to_mask(overlap_mask, first_borders)

        first_box = get_overlap(first, first_borders)
        second_box = get_overlap(patches[index], second_borders)
        summed = first_box + second_box

        first = sub_overlap(first, first_borders, summed)

        borders_list.append((first_borders, second_borders))

    first = first/overlap_mask

    for i in range(indexes.size):
        averages = get_overlap(first, borders_list[i][0])
        patches[indexes[i]] = sub_overlap(patches[indexes[i]], borders_list[i][1], averages)

    return first, patches

def add_to_mask(mask, borders):
    """
    Adds plus 1 to the mask values in voxels that overlap with another patch

    Parameters
    ----------
    mask : ndarray
        An array containing the times per voxel for the current first patch that overlapping values have been summed for that voxel

    borders : list
        A list containing the x-, y-, z-coordinates for the region in this patch that overlaps with another patch

    Returns
    -------
    mask : ndarray
        The mask array with the overlapping voxel values increased by one
    """

    mask[borders[0]:borders[1], borders[2]:borders[3], borders[4]:borders[5]] += 1
    return mask

def sub_overlap(patch, borders, summed):
    """
    Substitutes the overlapping region of this patch with the summed result

    Parameters
    ----------
    patch : ndarray
        The original patch-array without the substituted results

    borders : list
        A list containing the x-, y-, z-coordinates for the region in this patch that overlaps with another patch

    summed : ndarray
        The summed results from two overlapping regions that is surrounded by the borders

    Returns
    -------
    new : ndarray
        The original patch with the overlapping region substituted with the summed
    """
    patch[borders[0]:borders[1], borders[2]:borders[3], borders[4]:borders[5]] = summed
    return patch

def get_overlap(patch, borders):
    """
    Gets the region in patch that overlaps with the second patch

    Parameters
    ----------
    patch : ndarray
        An array containing one patch proposed by the neural network.

    borders : list
        A list containing the x-, y-, z-coordinates for the region in this patch that overlaps with another patch

    Returns
    -------
    new : ndarray
        The region of this patch that overlaps with another patch
    """
    return patch[borders[0]:borders[1], borders[2]:borders[3], borders[4]:borders[5]]

def overlap_box(coordinate, borders, box_size):
    """
    Gets the relative coordinates for the box with the middle point in the coordinate. Can be used to extract wanted region from patch.
    The left corner of the box is in
        x - box_size/2
        y - box_size/2
        z - box_size/2

    Parameters
    ----------
    coordinate : ndarray
        The middle points of the box in space.

    borders : list
        The coordinates in image space in which two patches overlap.

    box_size : int
        Length of one side of the box around the center coordinates.

    Returns
    -------
    new : list
        A list containing the cropping coordinates for the overlapping region in the patches own relative space. 
    """
    low_x = int(borders[0] - (coordinate[0] - box_size/2))
    low_y = int(borders[2] - (coordinate[1] - box_size/2))
    low_z = int(borders[4] - (coordinate[2] - box_size/2))

    high_x = int(borders[1] - (coordinate[0] - box_size/2))
    high_y = int(borders[3] - (coordinate[1] - box_size/2))
    high_z = int(borders[5] - (coordinate[2] - box_size/2))

    return low_x, high_x, low_y, high_y, low_z, high_z

def overlap_coord(first, second, box_size):
    """
    Gets the x-, y- and z-values of space in which the two boxes overlap.

    Parameters
    ----------
    first : ndarray
        The middle point coordinates of the first voxel

    second : ndarray
        The middle point coordinates of the second voxel

    box_size : int
        Length of one side of the box around the center coordinates.

    Returns
    -------
    new : low_x, high_x, low_y, high_y, low_z, high_z
        The x-, y-, and z-values of space in which the two boxes overlap.
    """
    low_x = max(first[0] - box_size/2, second[0] - box_size/2)
    high_x = min((first[0] + box_size/2, second[0] + box_size/2))

    low_y = max(first[1] - box_size/2, second[1]  - box_size/2)
    high_y = min((first[1] + box_size/2, second[1]  + box_size/2))

    low_z = max(first[2] - box_size/2, second[2]  - box_size/2)
    high_z = min((first[2] + box_size/2, second[2]  + box_size/2))

    return low_x, high_x, low_y, high_y, low_z, high_z

def positive_matches(patches, coordinates, ground_truth, box_size):
    """
    Checks if any of the voxel values in patch is different from zero in ground truth.

    Parameters
    ----------
    patches : list
        A list containing the patches

    coordinates : list
        An array containing the coordinates for the center points of the patches in respective order to given patches.

    ground_truth : ndarray
        An array containing the ground truth location(s).

    Returns
    -------
    new : list
        A list of boolean values for each patch in patches. True if voxel value is different from zero in ground truth, False otherwise.
    """
    truth_list = []

    for i in range(len(patches)):
        lows, highs, loc = GetIndexes(ground_truth, coordinates[i], box_size)
        truth_region = Carve(lows, highs, ground_truth)

        ones = truth_region == 1
        not_zero = patches[i] > 0
        not_zero = not_zero[ones]

        boolean = np.any(not_zero)
        truth_list.append(boolean)

    return truth_list

# truth = niload("C:\\Koodia\\EnnustavaAivokuvantaminen\\Images\\10035\\10035\\aneurysms.nii.gz")
# #The middle point for this aneurysm is 354, 212, 64

# coordinates = np.array([[96, 96, 96],
#                         [64, 64, 64],
#                         [64, 64, 96],
#                         [364, 232, 64],
#                         [396, 200, 64],
#                         [380, 232, 64],
#                         [412, 232, 80],
#                         [200, 200, 200],
#                         [248, 200, 232],
#                         [200, 152, 200],
#                         [248, 152, 232]])

# patches_test = pickle.load(open('sample.p','rb'))
# box_size = 64

# #For testing
# testi1 = (patches_test[0][0:32, 0:32, 0:32] + patches_test[1][32:64, 32:64, 32:64] + patches_test[2][32:64, 32:64, 0:32]) / 3 
# testi2 = (patches_test[1][0:32, 0:64, 32:64] + patches_test[2][0:32, 0:64, 0:32]) / 2
# testi3 = (patches_test[0][0:32, 0:32, 32:64] + patches_test[2][32:64, 32:64, 32:64]) / 2
# testi4 = (patches_test[3][48:64, 0:32, 16:64] + patches_test[4][16:32, 32:64, 16:64] + patches_test[5][32:48, 0:32, 16:64] + patches_test[6][0:16, 0:32, 0:48]) / 4
# testi5 = (patches_test[3][48:64, 32:64, 16:64] + patches_test[5][32:48, 32:64, 16:64] + patches_test[6][0:16, 32:64, 0:48]) / 3 
# testi6 = (patches_test[4][32:48, 32:64, 16:64] + patches_test[5][48:64, 0:32, 16:64] + patches_test[6][16:32, 0:32, 0:48]) / 3 
# testi7 = (patches_test[5][48:64, 32:64, 16:64] + patches_test[6][16:32, 32:64, 0:48]) / 2 
# testi8 = (patches_test[7][48:64, 0:16, 32:64] + patches_test[8][0:16, 0:16, 0:32] + patches_test[9][48:64, 48:64, 32:64] + patches_test[10][0:16, 48:64, 0:32]) / 4  
# testi9 = patches_test[10][16:64, 0:48, 0:64]

# # Parameters: list of patches; length of one side of the box; a list of coordinates of middle points (first one for first patch and so on)
# averaged_patches = overlapping_patches(patches_test, box_size, coordinates)

# #For testing
# print("-----")
# print(np.allclose(averaged_patches[0][0:32, 0:32, 0:32], testi1, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[1][0:32, 0:64, 32:64], testi2, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[2][32:64, 32:64, 32:64], testi3, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[3][48:64, 0:32, 16:64], testi4, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[6][0:16, 32:64, 0:48], testi5, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[5][48:64, 0:32, 16:64], testi6, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[6][16:32, 32:64, 0:48], testi7, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[9][48:64, 48:64, 32:64], testi8, atol=1e-8))
# print("-----")
# print(np.allclose(averaged_patches[10][16:64, 0:48, 0:64], testi9, atol=1e-8))
# print("-----")

# # Parameters: list of patches; a list of coordinates of middle points (first one for first patch and so on); the ground truth ndarray, box side length
# boolean_list = positive_matches(averaged_patches, coordinates, truth, box_size) 

# print(boolean_list)