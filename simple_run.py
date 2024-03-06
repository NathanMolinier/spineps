import os
import re
import argparse
from pathlib import Path

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


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 database format.')
    parser.add_argument('--path-in', required=True, help='Path to the input image. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T2w.nii.gz (Required)')
    parser.add_argument('--ofolder', required=True, help='Path to the output directory. Example: ~/data/spineps-predictions (Required)')
    return parser


def main():
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
        "model_semantic": 't2w_segmentor',
        "model_instance": 'inst_vertebra',
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
        logger.print(f"Error with instance segmentation: errcode {errcode}")
    
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

    # Exctract discs labels

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