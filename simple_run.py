import os
import re
import argparse
from pathlib import Path
import cc3d
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cv2
import time

from image import Image, zeros_like

from spineps.phase_post import phase_postprocess_combined
from spineps.models import get_semantic_model, get_instance_model
from spineps.phase_pre import preprocess_input
from spineps.phase_semantic import predict_semantic_mask
from spineps.phase_instance import predict_instance_mask
from spineps.seg_enums import Acquisition, ErrCode, Modality
from spineps.seg_utils import (
    InputPackage,
    Modality_Pair,
    check_input_model_compatibility,
    check_model_modality_acquisition,
    find_best_matching_model,
)

from TPTBox.core.nii_wrapper import NII

DISCS_MAP = {2:1, 102: 3, 103: 4, 104: 5, 
             105: 6, 106: 7, 107: 8, 
             108: 9, 109: 10, 110: 11, 
             111: 12, 112: 13, 113: 14, 
             114: 15, 115: 16, 116: 17, 
             117: 18, 118: 19, 119: 20, 
             120: 21, 121: 22, 122: 23, 
             123: 24, 124: 25}

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Simplify SPINEPS cli.')
    parser.add_argument('--path-in', required=True, help='Path to the input image. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T2w.nii.gz (Required)')
    parser.add_argument('--ofolder', required=True, help='Path to the output directory. Example: ~/data/spineps-predictions (Required)')
    return parser


def main():
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    
    # Fetch paths
    img_path = args.path_in
    fname = os.path.basename(img_path)
    ofolder_path = args.ofolder

    # Load input image
    input_nii = NII.load(img_path, seg=False)
    input_package = InputPackage(
            input_nii,
            pad_size=4,
        )

    DEFAULTS = {
        "model_semantic": 't2w_segmentor_2.0',
        "model_instance": 'inst_vertebra_2.0',
        #
        "save_uncertainty_image": False,
        "save_softmax_logits": False,
        "save_debug_data": False,
        "save_modelres_mask": False,
        #
        "override_semantic": False,
        "override_instance": False,
        "override_postpair": False,
        "override_ctd": False,
        #
        "do_crop_semantic": True,
        "proc_n4correction": True,
        "ignore_compatibility_issues": False,
        "verbose": False,
        #
        "proc_fillholes": True,
        "proc_clean": True,
        "proc_corpus_clean": True,
        "proc_cleanvert": True,
        "proc_assign_missing_cc": True,
        "proc_largest_cc": 0
    }
    
    # Preprocess input
    debug_data_run: dict[str, NII] = {}
    input_preprocessed, errcode = preprocess_input(
                input_nii,
                pad_size=input_package.pad_size,
                debug_data=debug_data_run, 
                do_crop=DEFAULTS['do_crop_semantic'],
                do_n4=DEFAULTS['proc_n4correction'],
                verbose=DEFAULTS['verbose'],
            )
    # Run semantic segmentation
    seg_nii_modelres, unc_nii, softmax_logits, errcode = predict_semantic_mask(
                input_preprocessed,
                model=get_semantic_model(DEFAULTS['model_semantic']).load(),
                debug_data=debug_data_run,
                verbose=DEFAULTS['verbose'],
                fill_holes=DEFAULTS['proc_fillholes'],
                clean_artifacts=DEFAULTS['proc_clean'],
            )
    
    if (seg_nii_modelres.get_seg_array() == 0).all() or errcode != ErrCode.OK:
        raise ValueError(f"Error with semantic segmentation: errcode {errcode}")
    
    # Save semantic prediction
    out_spine_raw = os.path.join(ofolder_path, get_mask_name_from_img_name(fname, suffix='_label-rawspine_dseg'))
    seg_nii_modelres.save(out_spine_raw)

    # Run instance prediction
    whole_vert_nii, errcode = predict_instance_mask(
                seg_nii_modelres.copy(),
                model=get_instance_model(DEFAULTS['model_instance']).load(),
                debug_data=debug_data_run,
                verbose=DEFAULTS['verbose'],
                fill_holes=DEFAULTS['proc_fillholes'],
                proc_corpus_clean=DEFAULTS['proc_corpus_clean'],
                proc_cleanvert=DEFAULTS['proc_cleanvert'],
                proc_largest_cc=DEFAULTS['proc_largest_cc'],
            )

    # Check for errors with instance segmentation
    if errcode != ErrCode.OK:
        print(f"Error with instance segmentation: errcode {errcode}")
    
    # Save instance prediction
    out_vert_raw = os.path.join(ofolder_path, get_mask_name_from_img_name(fname, suffix='_label-rawvert_dseg'))
    whole_vert_nii.save(out_vert_raw)

    # Cleanup Steps
    seg_nii_back = input_package.sample_to_this(seg_nii_modelres)
    whole_vert_nii = input_package.sample_to_this(whole_vert_nii, intermediate_nii=seg_nii_modelres)

    # use both seg_raw and vert_raw to clean each other, add ivd_ep ...
    seg_nii_clean, vert_nii_clean = phase_postprocess_combined(
        seg_nii=seg_nii_back,
        vert_nii=whole_vert_nii,
        debug_data=debug_data_run,
        labeling_offset=1,
        proc_assign_missing_cc=DEFAULTS['proc_assign_missing_cc'],
        verbose=DEFAULTS['verbose'],
    )

    seg_nii_clean.assert_affine(shape=vert_nii_clean.shape, zoom=vert_nii_clean.zoom, orientation=vert_nii_clean.orientation)
    input_package.make_nii_from_this(seg_nii_clean)
    input_package.make_nii_from_this(vert_nii_clean)

    # Save cleaned images
    out_spine = os.path.join(ofolder_path, get_mask_name_from_img_name(fname, suffix='_label-spine_dseg'))
    out_vert = os.path.join(ofolder_path, get_mask_name_from_img_name(fname, suffix='_label-vert_dseg'))
    seg_nii_clean.save(out_spine)
    vert_nii_clean.save(out_vert)

    # Extract discs labels
    vert_image = Image(out_vert)
    img = Image(img_path)
    discs_nii_clean = extract_discs_label(img, vert_image, ofolder_path, mapping=DISCS_MAP)

    # Save discs labels
    out_discs = os.path.join(ofolder_path, get_mask_name_from_img_name(fname, suffix='_label-discs_dlabel'))
    discs_nii_clean.save(out_discs)
    end_time = time.time()
    print('-'*20)
    print(f'Total time used for computation {end_time-start_time} seconds')
    print('-'*20)


##
def extract_discs_label(img, label, ofolder_path, mapping):
    print('Creating discs labels')
    orig_orientation = label.orientation

    # Use RSP orientation
    label.change_orientation('RSP')
    img.change_orientation('RSP')

    # Extract only discs segmentations based on mapping
    data = label.data
    data_discs_seg = np.zeros_like(data)
    for seg_value, discs_value in mapping.items():
        data_discs_seg[np.where(data==seg_value)] = discs_value
    
    # Deal with disc 1 obtained with the first vertebrae (Highest vertical coordinate)
    if 1 in data_discs_seg: 
        # If the first vertebrae is present identify label disc 1 at the top
        vert1_seg = np.array(np.where(data_discs_seg==1))
        disc1_idx = np.argmin(vert1_seg[1]) # find min on the S-I axis
        disc1_coord = vert1_seg[:,disc1_idx]
        data_discs_seg[np.where(data_discs_seg==1)] = 0
        data_discs_seg[disc1_coord[0], disc1_coord[1], disc1_coord[2]] = 1
    
    ## Identify the posterior tip of the disc
    # Extract the center of mass of every discs segmentation to create discs labels
    discs_centroids, discs_bb = extract_centroids_3D(data_discs_seg) # Centroids are sorted based on the vertical axis

    # Generate a centerline between the discs by doing linear interpolation
    yvals = np.linspace(discs_centroids[0, 1], discs_centroids[-1, 1], round(8*len(discs_centroids))) # TODO: Should we calculate the number of dots based on the resolution ?
    xvals = np.interp(yvals, discs_centroids[:,1], discs_centroids[:,0])
    zvals = np.interp(yvals, discs_centroids[:,1], discs_centroids[:,2])
    centerline = np.concatenate((np.expand_dims(xvals, axis=1), np.expand_dims(yvals, axis=1), np.expand_dims(zvals, axis=1)), axis=1)

    # Shift the centerline to the posterior direction until there is no intersection with the discs segmentations
    min_seg_AP = np.min(np.where(data_discs_seg>0)[2]) # Find the min coordinate of the discs segmentation on the A-P axis
    max_centroid_AP = np.max(discs_centroids[:,2])
    offset = 5
    shift = (max_centroid_AP - min_seg_AP + offset) if min_seg_AP >= offset else (max_centroid_AP - min_seg_AP)
    centerline_shifted = np.copy(centerline)
    centerline_shifted[:,2] = centerline_shifted[:,2] - shift

    # For each segmented disc, identify the closest voxel to this shifted centerline
    discs_list = closest_point_seg_to_line(data_discs_seg, centerline_shifted, discs_bb)

    # Add disc 2 between disc 1 and 3
    disc1_coord = discs_list[discs_list[:,-1]==1]
    disc2_coord = discs_list[discs_list[:,-1]==3]
    disc2_coord[0,1] = (disc2_coord[0,1] + disc1_coord[0,1])//2
    disc2_coord[0,-1] = 2
    discs_list = np.insert(discs_list, 1, disc2_coord, axis=0)

    # Create image plot
    shape = img.data.shape
    out_cv2 = np.zeros(data.shape[1:] + (3,)) # BGR
    out_cv2[:,:,0] = np.where(np.sum(data_discs_seg, axis=0)>0,1,0)*255 + create_2DGaussian_from_labels(centerline[:,1:], shape=shape[1:], radius=1)*178 + create_2DGaussian_from_labels(centerline_shifted[:,1:], shape=shape[1:], radius=1)*0 + create_2DGaussian_from_labels(discs_list[:,1:-1], shape=shape[1:], radius=3)*0 # B
    out_cv2[:,:,1] = np.where(np.sum(data_discs_seg, axis=0)>0,1,0)*255 + create_2DGaussian_from_labels(centerline[:,1:], shape=shape[1:], radius=1)*102 + create_2DGaussian_from_labels(centerline_shifted[:,1:], shape=shape[1:], radius=1)*0 + create_2DGaussian_from_labels(discs_list[:,1:-1], shape=shape[1:], radius=3)*0 # G
    out_cv2[:,:,2] = np.where(np.sum(data_discs_seg, axis=0)>0,1,0)*51 + create_2DGaussian_from_labels(centerline[:,1:], shape=shape[1:], radius=1)*255 + create_2DGaussian_from_labels(centerline_shifted[:,1:], shape=shape[1:], radius=1)*255 + create_2DGaussian_from_labels(discs_list[:,1:-1], shape=shape[1:], radius=3)*255 # R
    cv2.imwrite(os.path.join(ofolder_path,'pred_discs.png'), out_cv2)
    out_cv2 = np.zeros(data.shape[1:] + (3,)) # BGR
    out_cv2[:,:,0] = normalize(img.data[shape[0]//2, :, :])*255 - create_2DGaussian_from_labels(discs_list[:,1:-1], shape=shape[1:], radius=3)*255
    out_cv2[:,:,1] = normalize(img.data[shape[0]//2, :, :])*255 - create_2DGaussian_from_labels(discs_list[:,1:-1], shape=shape[1:], radius=3)*255
    out_cv2[:,:,2] = normalize(img.data[shape[0]//2, :, :])*255 - create_2DGaussian_from_labels(discs_list[:,1:-1], shape=shape[1:], radius=3)*0
    cv2.imwrite(os.path.join(ofolder_path,'output.png'), out_cv2)

    # Create output Image
    data_discs = np.zeros_like(data)
    for x, y, z, v in discs_list:
        data_discs[x, y, z] = v
    label.data = data_discs
    return label.change_orientation(orig_orientation)

def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi))

def project_point_on_line(point, line):
    """
    Project the input point on the referenced line by finding the minimal distance

    :param point: coordinates of a point and its value: point = numpy.array([x y z])
    :param line: list of points coordinates which composes the line
    :returns: closest coordinate to the referenced point on the line: projected_point = numpy.array([X Y Z])
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox
    """
    # Calculate distances between the referenced point and the line then keep the closest point
    dist = np.sum((line - point) ** 2, axis=1)

    return line[np.argmin(dist)], np.min(dist)

def closest_point_seg_to_line(discs_seg, centerline, bounding_boxes):
    """
    """
    discs_list = []
    for x, y, z in bounding_boxes:
        zer = np.zeros_like(discs_seg)
        zer[x, y, z] = discs_seg[x, y, z] # isolate disc
        # Loop on all the pixels of the segmentation
        min_dist = np.inf
        nonzero = np.where(zer>0)
        for u, v, w in zip(nonzero[0],nonzero[1],nonzero[2]):
            point, dist = project_point_on_line(np.array([u, v, w]), centerline)
            if dist < min_dist:
                min_dist = dist
                min_point = np.array([u, v, w, discs_seg[u, v, w]])
        discs_list.append(min_point)
    return np.array(discs_list)


def extract_centroids_3D(arr):
    '''
    Extract centroids and bouding boxes from a 3D numpy array
    :param arr: 3D numpy array
    '''
    stats = cc3d.statistics(cc3d.connected_components(arr))
    centroids = stats['centroids'][1:] # Remove backgroud <0>
    bounding_boxes = stats['bounding_boxes'][1:]
    sort_args = np.argsort(centroids[:,1]) # Sort according to the vertical axis because RSP orientation
    centroids_sorted = centroids[sort_args]
    bb_sorted = np.array(bounding_boxes)[sort_args]
    return centroids_sorted.astype(int), bb_sorted


def create_2DGaussian_from_labels(data, shape, c_dx=0, c_dy=0, radius=3):
    """
    Generate a Mask map from coordinates
    :param data : input image
    :param shape: dimension of output
    :param radius: is the radius of the gaussian function
    :param normalize : bool for normalization.
    :return: a MxN normalized array
    Based on https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    """
    img = np.zeros((shape))
    for coord in data:
        # Our 2-dimensional distribution will be over variables X and Y
        (M, N) = (shape[1], shape[0])

        x, y = coord[1], coord[0]

        # Add offset to the labels
        x += c_dx
        y += c_dy

        X = np.linspace(0, M - 1, M)
        Y = np.linspace(0, N - 1, N)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Mean vector and covariance matrix
        mu = np.array([x, y])
        Sigma = np.array([[radius, 0], [0, radius]])

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mu, Sigma)

        # Normalization
        Z *= (1 / np.max(Z))
        img += Z
    return img

def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N

def swap_ori_convention(ori):
    '''
    :param ori: input orientation (must be iterable)
    :return: swap orientation convention between min and max reading (e.g. RPI --> LAS)
    '''
    out_ori = ''
    for c in ori:
        if c == 'R':
            out_ori += 'L'
        elif c == 'L':
            out_ori += 'R'
        elif c == 'A':
            out_ori += 'P'
        elif c == 'P':
            out_ori += 'A'
        elif c == 'S':
            out_ori += 'I'
        elif c == 'I':
            out_ori += 'S'
        else:
            raise ValueError(f'Unknown value {c} used for orientation')
    
    if isinstance(ori, tuple):
        out_ori = tuple([c for c in out_ori])
    elif isinstance(ori, list):
        out_ori = [c for c in out_ori]
    elif not isinstance(ori, str):
        raise ValueError(f'Unknown format {type(ori)}')
    return out_ori

##
def get_mask_name_from_img_name(fname, suffix='_seg'):
    """
    This function returns the mask filename from an image filename.

    :param fname: Image fname
    :param suffix: Mask suffix
    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    """
    # Extract information from path
    subjectID, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(fname)

    # Extract file extension
    path_obj = Path(fname)
    ext = ''.join(path_obj.suffixes)

    # Create mask name
    mask_name = path_obj.name.split('.')[0] + suffix + ext

    return mask_name


##
def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    :return: contrast: MRI modality (dwi or anat)
    :return: echoID: echo ID (e.g., echo-1)
    :return: acquisition: acquisition (e.g., acq_sag)
    Copied from https://github.com/spinalcordtoolbox/manual-correction
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash

    echo = re.search('echo-(.*?)[_]', filename_path)     # [_/] means either underscore or slash
    echoID = echo.group(0)[:-1] if echo else ""    # [:-1] removes the last underscore or slash

    acq = re.search('acq-(.*?)[_]', filename_path)     # [_/] means either underscore or slash
    acquisition = acq.group(0)[:-1] if acq else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    contrast = 'dwi' if 'dwi' in filename_path else 'anat'  # Return contrast (dwi or anat)

    return subjectID, sessionID, filename, contrast, echoID, acquisition


if __name__=='__main__':
    main()