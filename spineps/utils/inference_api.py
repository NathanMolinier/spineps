import os
import os.path
import torch
from spineps.utils.predictor import nnUNetPredictor
from BIDS import NII, No_Logger, Log_Type
from BIDS.core import sitk_utils
import nibabel as nib
import numpy as np

logger = No_Logger()
logger.override_prefix = "API"


# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
def load_inf_model(
    model_folder: str,
    step_size: float = 0.5,
    ddevice: str = "cuda",
    use_folds: tuple[str, ...] | None = None,
    init_threads: bool = True,
    allow_non_final: bool = True,
    inference_augmentation: bool = False,
    verbose: bool = True,
) -> nnUNetPredictor:
    """Loades the Nako-Segmentor Model Predictor

    Args:
        model_folder (str, optional): nnUNet Result Model Folder containing the "fold_x" directories. Default to the basic folder
        step_size (float, optional): Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1.
        ddevice (str, optional): The device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID!. Defaults to "cuda".

    Returns:
        predictor: Loaded model predictor object
    """
    if ddevice == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count()) if init_threads else None
        device = torch.device("cpu")
    elif ddevice == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1) if init_threads else None
        torch.set_num_interop_threads(1) if init_threads else None
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    assert os.path.exists(model_folder), f"model-folder not found: got path {model_folder}"

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=inference_augmentation,  # <- mirroring augmentation!
        perform_everything_on_gpu=ddevice != "cpu",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
    )
    check_name = "checkpoint_final.pth"  # if not allow_non_final else "checkpoint_best.pth"
    try:
        predictor.initialize_from_trained_model_folder(model_folder, checkpoint_name=check_name, use_folds=use_folds)
    except Exception as e:
        if allow_non_final:
            predictor.initialize_from_trained_model_folder(model_folder, checkpoint_name="checkpoint_best.pth", use_folds=use_folds)
            logger.print("Checkpoint final not found, will load from best instead", Log_Type.WARNING)
        else:
            raise e
    logger.print(f"Inference Model loaded from {model_folder}") if verbose else None
    return predictor


def run_inference(
    input: str | NII | list[NII],
    predictor: nnUNetPredictor,
    # pad_size: int = 2,
    reorient_PIR: bool = False,
) -> tuple[NII, NII | None, np.ndarray]:
    """Runs nnUnet model inference on one input.

    Args:
        input (str | NII): Path to a nifty file or a NII object.
        predictor (_type_, optional): Loaded model predictor. If none, will load the default one. Defaults to None.

    Raises:
        AssertionError: If the input is not of expected type

    Returns:
        Segmentation (NII), Uncertainty Map (NII), Softmax Logits (numpy arr)
    """
    if isinstance(input, str):
        assert input.endswith(".nii.gz"), f"input file is not a .nii.gz! Got {input}"
        input = NII.load(input, seg=False)

    assert isinstance(input, NII) or isinstance(input, list), f"input must be a NII or str or list[NII], got {type(input)}"
    if isinstance(input, NII):
        input = [input]
    orientation = input[0].orientation
    header = input[0].header

    img_arrs = []
    # Prepare for nnUNet behavior
    for i in input:
        if reorient_PIR:
            i.reorient_()
        sitk_nii = sitk_utils.nii_to_sitk(i)
        nii_img_converted = i.get_array()
        # nii_img_converted = np.pad(nii_img_converted, pad_width=pad_size, mode="edge")
        nii_img_converted = np.swapaxes(nii_img_converted, 0, 2)[np.newaxis, :].astype(np.float16)
        img_arrs.append(nii_img_converted)
    affine = input[0].affine
    img = np.vstack(img_arrs)
    zoom = input[0].zoom
    props = {
        "sitk_stuff": {
            # this saves the sitk geometry information. This part is NOT used by nnU-Net!
            "spacing": sitk_nii.GetSpacing(),  # type:ignore
            "origin": sitk_nii.GetOrigin(),  # type:ignore
            "direction": sitk_nii.GetDirection(),  # type:ignore
        },
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
        # are returned x,y,z but spacing is returned z,y,x. Duh.
        "spacing": [zoom[2], zoom[1], zoom[0]],  # PIR
    }
    segmentation, seg_stacked, softmax_logits = predictor.predict_single_npy_array(  # type:ignore
        img,
        props,
        save_or_return_probabilities=True,
    )  # type:ignore
    del seg_stacked
    segmentation = np.swapaxes(segmentation, 0, 2)
    # segmentation[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
    # uncertainty
    # softmax_logits shape (label, R, I, P)
    softmax_logits = np.swapaxes(softmax_logits, 0, 3)
    # PRI label
    softmax_logits = np.swapaxes(softmax_logits, 1, 2)
    uncertainty_arr = np.max(softmax_logits, axis=-1)  # max of (average softmax from the different folds)
    # uncertainty_arr = np.swapaxes(uncertainty_arr, 0, 2)

    assert isinstance(segmentation, np.ndarray)
    seg_nii = NII(nib.ni1.Nifti1Image(segmentation, affine=affine, header=header), seg=True)

    uncertainty_nii = NII(
        nib.ni1.Nifti1Image(uncertainty_arr, affine=affine, header=header),
        seg=False,
    )

    seg_nii.reorient_(orientation)
    uncertainty_nii.reorient_(orientation)

    if np.isnan(uncertainty_nii.min()):
        return seg_nii, None, softmax_logits
    return seg_nii, uncertainty_nii, softmax_logits