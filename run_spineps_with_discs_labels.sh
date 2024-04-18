
# Function
label_with_spineps(){
    local img_path=$(realpath "$1")
    local out_path="$2"
    local contrast="$3"
    local img_name="$(basename "$img_path")"
    (
        # Create temporary directory
        tmpdir="$(mktemp -d)"
        echo "$tmpdir" was created

        # Copy image to temporary directory
        tmp_img_path="${tmpdir}/${img_name}"
        cp "$img_path" "$tmp_img_path"

        # Activate conda env
        eval "$(conda shell.bash hook)"
        conda activate spineps

        # Select semantic weights
        if [ "$contrast" = "t1" ];
            then semantic=t1w_segmentor;
            else semantic=t2w_segmentor_2.0;
        fi
        
        # Run SPINEPS on image
        spineps sample -i "$tmp_img_path" -model_semantic "$semantic" -model_instance inst_vertebra_2.0 -dn derivatives
        
        # Run vertebral labeling with SPINEPS vertebrae prediction
        vert_path="$(echo ${tmpdir}/derivatives/*_seg-vert_msk.nii.gz)"
        python3 generate_discs_labels.py --path-vert "$vert_path" --path-out "$out_path"

        # Remove temporary directory
        rm -r "$tmpdir"
        echo "$tmpdir" was removed

        # Deactivate conda environment
        conda deactivate
    )
}

IMG_PATH="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-spineps/data-multi-subject/sub-amu01_T2w.nii.gz"
OUT_PATH="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-spineps/data-multi-subject/test/sub-amu01_T2w_label-vert_dlabel.nii.gz"
CONTRAST="t2"

label_with_spineps "$IMG_PATH" "$OUT_PATH" "$CONTRAST"