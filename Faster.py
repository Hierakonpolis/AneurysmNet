import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from operator import itemgetter
import time
from pathlib import Path
import math
import re
 
np.set_printoptions(precision=2, suppress=True)

def check_result(results, locations, box_size):
    """
    Checks if the received regions of interest contain the true aneurysms. Return true if all aneurysms found, false othervise. For testing purposes.
 
    Parameters
    ----------
    results : list
        A list of the coordinates of the middle voxels for proposed boxes of interest
        [[x, y, z]
         [x, y, z]
         [x, y, z]]
 
    locations: list
        A list of the location coordinates of the aneurysms. Has 3 values if one aneurysm exists, 6 if two, 9 if three and so on
        [1, 2, 3, 4, 5, 6, ..., n], so that n % 3 == 0

    box_size : int
        The length of one side of the box in which the aneurysm must be located in.
 
    Returns
    -------
    new : True or False
        Returns true if all aneurysms are found within the proposed boxes
        Returns false otherwise

    new : int
        Number of aneurysms found

    new : int
        Number of aneurysms missed
    """
    truth_list = []
    found = False
    missed = 0
    found_amount = 0

    if len(locations) == 0:
        return True, 0, 0

    nro_aneur = int(len(locations) / 3)

    for j in range(nro_aneur):
        for i in range(len(results)):
            if(abs(results[i][0] - locations[j + (j*2)]) < box_size/2):
                if(abs(results[i][1] - locations[j+1+ (j*2)]) < box_size/2):
                     if(abs(results[i][2] - locations[j+2+ (j*2)]) < box_size/2):
                         found = True
                         break
        truth_list.append(found)
        found = False

    if all(truth_list):
        for boolean in truth_list:
            found_amount += 1
        return True, found_amount, missed
    else: 
        for boolean in truth_list:
            if boolean:
                found_amount += 1
            else:
                missed += 1
        return False, found_amount, missed                              

def mass_middle(lista):
    """
    Calculates the middle point of the voxels in the array.
 
    Parameters
    ----------
    lista : ndarray
        An array containing the coordinations of voxels.
 
    Returns
    -------
    new : list
        A list containing the calculated middle voxels for the ndarray.
    """
    middles = ([[]])

    for i in range(len(lista)):
        middle = np.mean(lista[i], axis=0)
        middles.append(middle)

    return middles[1:]

def reg_of_int(lista, pix_close):
    """
    Divides the values to smaller groups within a given x-,  y- and z-axis distance. Faster than reg_of_int_neighbours - function, but the amount of proposed regions
    is higher.
    
    Principle:
        Takes the first coordinates from the ndarray and finds all the voxels close to this within the limits. Turns these voxel-coordinates into one group and removes
            them from the list containing all the voxels.
        Repeats until no voxels are left in the original list.

    Parameters
    ----------
    lista : ndarray
        An array containing the coordinates of voxels

    pix_close : int
        How close the voxels must be in x-, y- and z-axis to the voxel in consideration to be added to the same group. 

    Returns
    -------
    new : list
        A list containing ndarrays of groups of close voxels
    """

    all_groups = ([[]])
    values = lista.T

    while values.size > 3:
        group, values = np.split(values, [1], 1)

        indexes = np.nonzero(abs(values[:1] - group[0]) <= pix_close)
        indexes2 = np.nonzero(abs(values[1:2] - group[1]) <= pix_close)
        indexes3 = np.nonzero(abs(values[2:3] - group[2]) <= pix_close)

        indexes = indexes[1][np.isin(indexes[1], indexes2[1])]
        indexes = indexes[np.isin(indexes, indexes3[1])]

        if indexes.size > 0:
            close = values[:, indexes]
            values = np.delete(values, [indexes], 1)
            group = np.concatenate((group, close), axis=1)

        all_groups.append(group.T)

    if values.size == 3:
        all_groups.append(values.T)

    return all_groups[1:]

def reg_of_int_neighbours(lista, box_size, pix_close):
    """
    Divides the values to smaller groups within a given x-,  y- and z-axis distance. This function is slower than reg_of_int, especially as n grows,
    but produces fewer region proposals.
    
    Principle:
        Takes the first coordinates from the ndarray and finds all the voxels close to this within the limits. Repeats for all the added voxels. 
            This way the region contains a group of voxels for which every voxel is within the pix_close - distance of at least one other voxel.
            Turns these voxel-coordinates into one group and removes them from the list containing all the voxels.
        Repeats this until no voxels are left in the original list.

    Parameters
    ----------
    lista : ndarray
        An array containing the coordinates of voxels

    box_size : int
        The length of one side of the box in which the aneurysm must be located in.

    pix_close : int
        How close the voxels must be in x-, y- and z-axis to the voxel in consideration to be added to the same group. 

    Returns
    -------
    new : list
        A list containing ndarrays of groups of close voxels
    """
    all_groups = ([[]])
    values = lista.T
    first = np.array([0])

    while values.size > 3:
        
        anchor = values[:, first]
        all_indexes = closest_recursion(first, first, values, pix_close, anchor, box_size)

        if all_indexes is not None:
            close = values[:, all_indexes]
            values = np.delete(values, [all_indexes], 1)
        else:
            close = values[:, first]
            values = np.delete(values, first, 1)

        all_groups.append(close.T)

    if values.size == 3:
        all_groups.append(values.T)

    return all_groups[1:]

def closest_recursion(close_indexes, index, all_values, pix_close, anchor, box_size):
    """
    A recursive call used in reg_of_int_neighbours. Finds the indexes of all the voxels that are within the pix_close - value of each other.

    Parameters
    ----------
    close_indexes : ndarray
        Contains the indexes for all the voxels close to the index

    index : nro
        The index of the value in the all_values for which the neighbouring voxels are being searched for

    all_values : ndarray
        An array containing all the coordinate values for the voxels

    pix_close : int
        How close the voxels must be in x-, y- and z-axis to the voxel in consideration to be added to the same group.

    Returns
    -------
    new : ndarray
        The indixes of voxels close to the index.

    """
    if all_values.size == 0:
        return close_indexes

    group = all_values[:, index]
    values = all_values

    indexes = np.nonzero(abs(values[:1] - group[0]) <= pix_close)
    indexes2 = np.nonzero(abs(values[1:2] - group[1]) <= pix_close)
    indexes3 = np.nonzero(abs(values[2:3] - group[2]) <= pix_close)

    indexes = indexes[1][np.isin(indexes[1], indexes2[1])]
    indexes = indexes[np.isin(indexes, indexes3[1])]

    indexes_bool = np.invert(np.isin(indexes, close_indexes))
    indexes = indexes[indexes_bool]
    
    if len(indexes) == 0:
        return None

    close_indexes = np.append(close_indexes, indexes)

    for index in indexes:
        if abs(values[:, index][0] - anchor[0]) < box_size/2 and abs(values[:, index][1] - anchor[1]) < box_size/2 and abs(values[:, index][2] - anchor[2]) < box_size/2:
            more = closest_recursion(close_indexes, np.array([index]), values, pix_close, anchor, box_size)
            if more is not None:
                close_indexes = np.append(close_indexes, more)
                close_indexes = np.unique(close_indexes)

    close_indexes = np.unique(close_indexes)
    return close_indexes

#def inside_box(values, group, indexes, box_size):
#    TODO

def top_values(data, prc):
    """
    Returns the top percentage highest values from provided data.
 
    Parameters
    ----------
    data : numpy.ndarray
        Nifti volume as a data array
 
    pct : float
        The percentage of top values wanted
 
    Returns
    -------
    new : numpy.ndarray
        Numpy-volume containing the top percentage highest values
    """
    size = int(np.prod(data.shape) * (1 - prc))
    
    bright_ind = np.argpartition(data, size, None)[size:]

    bright_co = np.unravel_index(bright_ind, data.shape)

    bright_co_np = np.array([bright_co[0], bright_co[1], bright_co[2]]).T

    return bright_co_np

def potential_aneurysm(data, box_size, accuracy, neighbours, pix_close):
    """
    Goes through the given image and returns a list of possible aneurysms
 
    Parameters
    ----------
    data : numpy.ndarray
        Data array of the MRA image of interest
 
    box_size : int
        Size of one side of the box used in segmenting potential aneurysm sites

    accuracy: float
        The top percentage of high value voxels taken into consideration

    nro : int
        The number of regions the image will be split to. Speeds up the algorithm.
 
    Returns
    -------
    new : list
        A list that has the middle voxels coordinates of all the possible aneurysms proposed by this algorithm
 
        [[x, y, z]
         [x, y, z]
         [x, y, z]]
    """
    list1 = top_values(data, accuracy)

    list1 = list1[list1[:,2].argsort()]
    list1 = list1[list1[:,1].argsort(kind='stable')]
    list1 = list1[list1[:,0].argsort(kind='stable')]
 
    list2 = []
    list3 = []

    if neighbours:
        list2.append(reg_of_int_neighbours(list1, box_size, pix_close))
    else:
        list2.append(reg_of_int(list1, pix_close))

    for j in range(0, len(list2)):
        list3.append(mass_middle(list2[j]))

    list3 = [y for x in list3 for y in x]

    return list3

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

def data_locations(location_text, numeric):
    """
    Reads the textfile and fetches the wanted information from the file.

    Parameters
    ----------
    locations_text : string
	    Path to the text-file, containing the locations of the images or aneurysms. One location or the locations of aneurysms from one image

    numeric : boolean
	    True, if the location text-file contains the aneuryms coordinates
	    False, if location text-file contains the paths to images

    Returns
    -------
    new : list
        A list containing the locations of images or the coordinates of the aneurysms. One value per line
    """
    file = open(location_text)
    lines = file.read().splitlines()

    if numeric:
        for i in range(len(lines)):
            lines[i] = lines[i].split(',')
            if len(lines[i]) == 1:
                lines[i] = ''
                continue
            lines[i] = [int(j) for j in lines[i]]

    return lines

def write_aneurysm_co(locations_text):
    """
    Produces a text-file containing the numerical locations for each aneurysm

    Parameters
    ----------
    locations_text : Path
        A Path-object containing the path to the text-file containing the locations to aneurysm-location text-files

    Returns
    -------
    new : Path
        Returns a Path-object to the text file containing the aneurysm coordinates for each MRA-image.
    """

    data_file = open(locations_text)
    data = data_file.read().splitlines()
    data_file.close()
    all = []

    for i in range(len(data)):
        data_file = open(data[i])
        lines = data_file.read().splitlines()
        if len(lines) > 0:
            for j in range(len(lines)):
                index = lines[j].rfind(",")
                lines[j] = lines[j][:index]
                lines[j].replace(" ", "")    
        all.append(lines)
        data_file.close()

    data_file = open("kaikki.txt", 'w')

    for k in range(len(all)):
        s1 = ','.join(all[k])
        data_file.write(s1 + '\n')

    data_file.close()

    location = Path(os.getcwd()) / "kaikki.txt"
    return location

def make_location_txt(directory, numeric):
    """
    Produces a text file containing the locations of MRA-images or aneurysm-location text-files to the current working directory

    Parameters
    ----------
    directory : string
        A string representation for the path to ADAM_release_subjs-folder

    numeric : boolean
        True: text file contains the locations of aneurysm-location text-files
        False: text file contains the locations of MRA-images

    Returns
    -------
    new : Path
        Returns a Path-object to the text file containing produced information.

    """

    path = Path(directory)

    list_of_files = []

    if numeric:
        name = "location.txt"
        file_name = "Aneurysms.txt"
    else:
        name = "TOF.nii.gz"
        file_name = "Locations.txt"

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(name):
                list_of_files.append(os.path.join(subdir, file))

    for j in reversed(range(len(list_of_files))):
        if "pre" in list_of_files[j]:
            del list_of_files[j]

    data_file = open(file_name, 'w')

    for i in range(len(list_of_files)):
        data_file.write(list_of_files[i] + '\n')

    data_file.close()

    location = Path(os.getcwd()) / file_name
    return location

def get_name(whole_name):
    """
    Returns the folders number name, for example 10039F

    Parameters
    ----------
    whole_name : string
        A string of the location of the image

    Returns
    -------
    new : string
        The folders number name
    """

    regex = re.compile(r'(\d\d\d\d\d[A-Z]?)')

    name = regex.search(whole_name)
    name = name.group()
    return name

def run_tests(box_size, accuracy, neighbours, pix_close, directory):
    """
    Run tests for different parameters. 

    Parameters
    ----------

    box_size : int
        Length of one side of a box of interest. Used to test whether the aneurysm is found within the box around one of the middle points .

    accuracy : float
	    The top percentage of brightest voxels to be included.

    neighbours : boolean   USE ONLY FALSE AT THE MOMENT!!!
        Determines the function used in grouping the voxels.
        True: The reg_of_int_neighbours - function is used. See function for details
        False: The reg_of_int - function is used. See function for details

    pix_close : int
        Used to determine how close within x-, y- and z-axis the points must be. Algorithm works faster with larger values, but the proposals are not
        as accurate.

    directory : string
        A string representation for the path to ADAM_release_subjs-folder
    """
    path_to_MRA = make_location_txt(directory, False)
    path_to_aneu = make_location_txt(directory, True)

    path_to_aneu = write_aneurysm_co(path_to_aneu)

    MRA_images = data_locations(path_to_MRA, False)
    aneurysm_locations = data_locations(path_to_aneu, True)

    found_nro = 0
    found_total = 0
    not_found_total = 0
    not_found_one = 0
    end_ind = int(len(MRA_images))
    start_ind = 0
    amount = int(end_ind - start_ind)
    list_not_found = []

    # print('')
    # print('--START OF RUN--')
    start_time = time.time()
    results_sum = 0


    for i in range(start_ind, end_ind):
        data = niload(Path(MRA_images[i]))
        
        dataset_name = get_name(MRA_images[i])

        results = potential_aneurysm(data, box_size, accuracy, neighbours, pix_close) # Gets the results

        passed = time.time() - start_time

        found, found_one, not_found_one = check_result(results, aneurysm_locations[i], box_size) # Checks whether the aneurysms are in at least one of the proposed regions 

        if found:
            # print("Aneurysms found from image nro", dataset_name + '.', "Time:", str(passed), "seconds" + '.', 'Number of regions proposed: ', str(len(results)))
            found_nro += 1
            results_sum += len(results)
            found_total += found_one
        else:
            # print("Aneurysms not found from image", dataset_name + '.', "Time:", str(passed), "seconds" + '.', 'Number of regions proposed: ', str(len(results)))
            list_not_found.append(('From dataset: ' + str(dataset_name) + '. Nro of aneurysms not found: ' + str(not_found_one)))
            results_sum += len(results)
            not_found_total += not_found_one
            found_total += found_one

 
    # print('-----')
    # print("Number of datasets went through:", str(amount), ", from which the aneurysms were found in", str(found_nro))
    # print('-----')
    # if len(list_not_found) > 0:
    #     print('--ANEURYSMS NOT FOUND--')
    #     print('Total number of aneurysms:', str(not_found_total + found_total))
    #     print('Total number of aneurysms not found:', str(not_found_total))
    #     print('Percentage missed:' + str(not_found_total/found_total))
    #     for j in range(len(list_not_found)):
    #         print(list_not_found[j])
    #     print('-----')
    # else:
    #     print('Total number of aneurysms:', str(not_found_total + found_total))

    # print("Total time: ", str(passed), "seconds")
    # print("Time per image: ", str(passed/amount), "seconds")
    # print("The average of proposed regions per image: ", str(results_sum / amount))

    # print('-----')
    # print('Used parameters:')
    # print('Box-size: ', box_size)
    # print('Percentage: ', accuracy)
    # print('Neighbours: ', neighbours)
    # print('Pixel closeness value:', pix_close)
    # print('--END OF RUN--')
    # print('')
    
    return (not_found_total,results_sum, box_size, accuracy, neighbours, pix_close)

# You can check the functions commentations for the parameters.
# run_tests(64, 0.001, False, 10, '/media/Olowoo/ADAM_release_subjs')
run_tests(64, 0.005, False, 12, '/media/Olowoo/ADAM_release_subjs')

percs=[0.005,0.002,0.001,0.0005]
dist=[3,5,10,15,20]
total=len(percs)*len(dist)
resultlist=[]
processes=[]

for percentile in percs:
    for distdelta in dist:
        for theothervariable in [True,False]:
            resultlist.append(run_tests(64, percentile, theothervariable, distdelta,'/scratch/project_2003143/ADAM_release_subjs'))

import pickle
pickle.dump(resultlist,open('/projappl/project_2003143/PatchGridSearch.p','wb'))
        
