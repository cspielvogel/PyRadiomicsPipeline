#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mar 15 16:24 2021

Pyradiomics-based image feature extraction pipeline

# TODO: test dilated vs non-dilated segmentation

@author: cspielvogel
"""

from __future__ import print_function

import logging
import os
import time

import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom

import radiomics
from radiomics import featureextractor

import nrrd


def convert_floats_to_ranks(floats):
    """Convert list of float values to list of integer values starting from 0 while maintaining the order of values"""
    if isinstance(floats[0], str):
        ordered_unique_floats = sorted([float(val.replace(",", ".")) for val in np.unique(floats)])
    else:  # Formatting in merged-lesion files
        ordered_unique_floats = sorted([float(val) for val in np.unique(floats)])

    ref = [np.round(float(val), 2) for val in ordered_unique_floats]

    if isinstance(floats[0], str):
        return [ref.index(np.round(float(val.replace(",", ".")), 2)) for val in floats]
    else:  # Formatting in merged-lesion files
        return [ref.index(np.round(float(val), 2)) for val in floats]


def npy_to_nrrd(npy, output_path):
    """Saves the given NPY array as NRRD file in the given output path"""
    nrrd.write(output_path, npy)


def mm_to_npy(input_path):
    """
    Load mm coordinate file from input path and convert to numpy array.
    """

    # Check if file is a merged-lesion file (due to its different formatting)
    merged = True if "/Lesion-Merged/" in input_path else False

    # Read mm coordinate file
    if merged:
        mm = pd.read_csv(input_path, sep=";", skiprows=2)
    else:
        mm = pd.read_csv(input_path, sep=";", skiprows=5)

    # Convert mm coordinates to rank indices
    index_table = pd.DataFrame()

    if merged:
        index_table["x"] = convert_floats_to_ranks(mm["x"].values)
        index_table["y"] = convert_floats_to_ranks(mm["y"].values)
        index_table["z"] = convert_floats_to_ranks(mm["z"].values)
        index_table["value"] = [float(val) for val in mm["Value"]]
    else:    # Merged-Lesion mm files do not have a unit in their column names
        index_table["x"] = convert_floats_to_ranks(mm["x(mm)"].values)
        index_table["y"] = convert_floats_to_ranks(mm["y(mm)"].values)
        index_table["z"] = convert_floats_to_ranks(mm["z(mm)"].values)
        index_table["value"] = [float(val.replace(",", ".")) for val in mm["Value"]]

    # Convert rank index table to NPY
    npy = np.zeros((np.max(index_table["x"]) + 1,
                    np.max(index_table["y"]) + 1,
                    np.max(index_table["z"]) + 1))

    for voxel_idx in index_table.index:
        voxel = index_table.iloc[voxel_idx]
        npy[int(voxel.x), int(voxel.y), int(voxel.z)] = voxel.value

    return npy


def bck_normalization(npy, lesion_path):
    """
    Takes a numpy array to be normalized and its path as input and returns the normalized numpy array.
    Normalization is conducted using the mean voxel value of the Background area in the Bck folder of the same scan as
    the lesion
    """

    # Derive background path from lesion path
    stem = "/".join(lesion_path.split("/")[:-3])
    bck_path = os.path.join(stem, "Bck/PET.csv")

    # Get background lesion
    bck = mm_to_npy(bck_path)

    # Return numpy array normalized with mean of background
    return np.round((npy / np.mean(bck)) * 1000).astype(int)


def mask_from_mm_npy(npy, threshold="mode"):
    """
    Take a NPY array derived from a mm coordinate file and create a mask from it.
    The VOI will contain any voxel with a value larger than the given threshold value.
    If the threshold value is "min", the threshold will automatically be set to the lowest value in the numpy array
    If the threshold is "mode", only the values with exactly the most common value will be excluded from the mask
    """
    if threshold == "min":
        threshold = np.min(npy)
    elif threshold == "mode":
        value, count = np.unique(npy, return_counts=True)
        threshold = value[np.argmax(count)]
        return (npy != threshold) * 1

    return (npy > threshold) * 1


def lesion_id(path):
    """Takes mm coordinate file path as input and returns patient ID, scan number and lesion number"""
    path_chunks = path.split("/Data")[2].split("/")

    # Check whether lesion (length 5) or background (length 4)
    if len(path_chunks) > 5:
        path_id = "{}-{}-{}-{}-{}".format(path_chunks[1],
                                          path_chunks[2],
                                          path_chunks[3],
                                          path_chunks[4],
                                          path_chunks[5].rstrip(".csv"))
    else:
        path_id = "{}-{}-{}-{}".format(path_chunks[1],
                                       path_chunks[2],
                                       path_chunks[3],
                                       path_chunks[4].rstrip(".csv"))
    return path_id


def get_mm_lesion_files(data_path, modality=None, get_bck=False):
    """Take project data path and return a list with all lesion paths"""
    mm_files = []
    for pat_id in os.listdir(data_path):
        patient_path = os.path.join(data_path, pat_id)
        for scan in os.listdir(patient_path):
            scan_path = os.path.join(patient_path, scan)
            for lesion in os.listdir(scan_path):

                # Get individual lesion folders
                if lesion.startswith("Lesion-") and not lesion.startswith("Lesion-Merged"):
                    lesion_path = os.path.join(scan_path, lesion)
                    voi_path = os.path.join(lesion_path, "Dilated")     # Replace VOIs to Dilated for adding
                    for mm_file in os.listdir(voi_path):
                        if mm_file.endswith(".csv"):
                            if modality is None:
                                mm_files.append(os.path.join(voi_path, mm_file))
                            elif modality == "PET":
                                if mm_file == "PET.csv":
                                    mm_files.append(os.path.join(voi_path, mm_file))
                            elif modality == "CT":
                                if mm_file == "CT.csv":
                                    mm_files.append(os.path.join(voi_path, mm_file))
                            elif modality == "MRI.csv":
                                if mm_file == "MRI.csv":
                                    mm_files.append(os.path.join(voi_path, mm_file))

                # Get background folder
                if get_bck is True and lesion == "Bck":
                    bck_path = os.path.join(scan_path, lesion)
                    mm_files.append(os.path.join(bck_path, "PET.csv"))

    return mm_files


def shift_range(npy, by=1024):
    """Add value specified by the by-parameter to all voxels in the numpy array"""
    return npy + by


def main():
    start = time.time()

    # Set paths
    # data_path = "/media/3atf_storage/3ATF/Clemens/Data/TestData/Data"
    data_path = "/media/3atf_storage/3ATF/Clemens/Data/HNSCC_AUT/Data"
    # mm_file_path = "/media/3atf_storage/3ATF/Clemens/Data/Cervical/Data/001/Scan-0/Lesion-0/Dilated/CT.csv"
    sample_list_path = "/home/cspielvogel/PycharmProjects/RadiomicsPipeline/Data/sample_list.csv"
    feature_output_path = "/home/cspielvogel/PycharmProjects/RadiomicsPipeline/Results/radiomics_features.csv"
    log_file_path = "/home/cspielvogel/PycharmProjects/RadiomicsPipeline/pyrad_log.txt"
    param_path = "/home/cspielvogel/PycharmProjects/RadiomicsPipeline/Settings/Params.yaml"

    # Get all mm coordinate files
    mm_paths = get_mm_lesion_files(data_path)[:10]
    # mm_paths = ["/media/3atf_storage/3ATF/Clemens/Data/HNSCC_AUT/Data/001/Scan-0/Lesion-0/Dilated/CT.csv",
    #             "/media/3atf_storage/3ATF/Clemens/Data/HNSCC_AUT/Data/001/Scan-0/Lesion-0/Dilated/PET.csv",
    #             "/media/3atf_storage/3ATF/Clemens/Data/HNSCC_AUT/Data/003/Scan-1/Lesion-0/Dilated/CT.csv",
    #             "/media/3atf_storage/3ATF/Clemens/Data/HNSCC_AUT/Data/003/Scan-1/Lesion-0/Dilated/PET.csv"]

    # Configure logging
    r_logger = logging.getLogger("radiomics")

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=log_file_path, mode="w")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    r_logger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = r_logger.getChild("batch")

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    # Create mask for mm coordinate files and save image and mask as NRRD files for each lesion
    images = []
    masks = []
    ids = []
    num_paths = len(mm_paths)
    for index, lesion_path in enumerate(mm_paths):

        # Create unique lesion ID
        current_lesion_id = lesion_id(lesion_path)
        ids.append(current_lesion_id)

        # Convert mm coordinate files to numpy arrays
        npy = mm_to_npy(lesion_path)

        # Shift Housfield units to be positive
        if lesion_path.endswith("/CT.csv"):
            npy = shift_range(npy, by=1024)

        # Perform background normalization for PET data to get TBR from bq/ml
        if "/PET.csv" in lesion_path:
            npy = bck_normalization(npy, lesion_path)

        # Save numpy arrays to NRRD files
        img_path = "/home/cspielvogel/PycharmProjects/RadiomicsPipeline/Data/{}-image.nrrd".format(current_lesion_id)
        npy_to_nrrd(npy, img_path)
        images.append(img_path)

        # Create masks
        if "/PET.csv" in lesion_path:   # For PET
            mask = mask_from_mm_npy(npy, threshold=0)
        else:
            mask = mask_from_mm_npy(npy, threshold="mode")

            # # For CT, by interpolating PET mask
            # pet_path = lesion_path.replace("/PET.csv", "/CT.csv")
            # pet = mm_to_npy(pet_path)
            # pet_mask = mask_from_mm_npy(npy)
            # mask_slices_interpolated = [zoom(pet_mask[:, :, i], 3) for i in np.arange(pet_mask.shape[2])]
            # np.dstack(mask_slices_interpolated)
        mask_path = "/home/cspielvogel/PycharmProjects/RadiomicsPipeline/Data/{}-mask.nrrd".format(current_lesion_id)
        npy_to_nrrd(mask, mask_path)
        masks.append(mask_path)

        # Display logging info
        print(lesion_path)
        logger.info("Loading mm file, normalization and mask creation completed for {}/{}".format(index+1, num_paths))

    # Create NRRD data table
    samples = pd.DataFrame()
    samples["Image"] = images
    samples["Mask"] = masks
    samples.index.name = "PAT.-ID"
    samples.index = ids
    samples.to_csv(sample_list_path, sep=";")

    logger.info("pyradiomics version: %s", radiomics.__version__)
    logger.info("Loading CSV")

    try:
        flists = pd.read_csv(sample_list_path, sep=";").T
    except Exception:
        logger.error("CSV READ FAILED", exc_info=True)
        exit(-1)

    logger.info("Loading Done")
    logger.info("Patients: %d", len(flists.columns))

    if os.path.isfile(param_path):
        extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
    else:
        raise FileNotFoundError("No settings file found at {}".format(param_path))

    logger.info("Enabled input images types: %s", extractor.enabledImagetypes)
    logger.info("Enabled features: %s", extractor.enabledFeatures)
    logger.info("Current settings: %s", extractor.settings)

    # Instantiate a pandas data frame to hold the results of all patients
    results = pd.DataFrame()

    # Iterate through samples
    for entry in flists:
        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                    entry + 1,
                    len(flists),
                    flists[entry]["Image"],
                    flists[entry]["Mask"])

        image_filepath = flists[entry]["Image"]
        mask_filepath = flists[entry]["Mask"]
        label = flists[entry].get("Label", None)

        if str(label).isdigit():
            label = int(label)
        else:
            label = None

        if (image_filepath is not None) and (mask_filepath is not None):
            feature_vector = flists[entry]    # This is a pandas Series
            feature_vector["Image"] = os.path.basename(image_filepath)
            feature_vector["Mask"] = os.path.basename(mask_filepath)

            try:
                result = pd.Series(extractor.execute(image_filepath, mask_filepath, label))
                feature_vector = feature_vector.append(result)
            except Exception:
                logger.error("FEATURE EXTRACTION FAILED:", exc_info=True)

            feature_vector.name = entry
            results = results.join(feature_vector, how="outer")

    # Remove diagnostic features and file paths from features
    for index in results.index:
        if index.startswith("diagnostics") or index == "Image" or index == "Mask":
            results = results.drop(index, axis="rows")

    # Add name to later index column
    index_list = results.index.tolist()
    index_list[0] = "ID"
    results.index = index_list

    # Find columns with features for the same instance and different modalities
    instance_id_to_index = {}
    for index, instance_id_raw in enumerate(results.loc["ID"]):
        instance_id = str(instance_id_raw).rstrip("-PET").rstrip("-CT").rstrip("-MRI")
        if str(instance_id_raw).endswith("-PET"):
            instance_id_to_index[instance_id][0] = index
        else:
            try:
                instance_id_to_index[instance_id][1] = index
            except KeyError:
                instance_id_to_index[instance_id] = [None, index]

    # Remove ID columns from results table
    results = results.drop("ID", axis="rows")

    # Create results table with one column per instance and one row for each feature per modality
    results_multimodality = pd.DataFrame(np.append([value + "-PET" for value in results.index],
                                                   [value + "-CT" for value in results.index]))

    for instance_id in instance_id_to_index.keys():
        pet_index = instance_id_to_index[instance_id][0]
        ct_index = instance_id_to_index[instance_id][1]
        results_multimodality[instance_id] = np.append(results[results.columns[pet_index]],
                                                       results[results.columns[ct_index]])

    # Write results to CSV file
    logger.info("Extraction complete, writing CSV to {}".format(feature_output_path))
    # results_multimodality.set_index("ID", inplace=True)
    results_multimodality.to_csv(feature_output_path, index=False, na_rep="NaN")
    logger.info("CSV writing complete")

    # Display elapsed time
    print("Total time (mins):", (time.time() - start) / 60)


if __name__ == "__main__":
    main()
