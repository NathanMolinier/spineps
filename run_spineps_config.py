import os
import json
import argparse
from simple_run import run_prediction
import subprocess

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cGAN and SPINEPS inference on a JSON config file')
    parser.add_argument('--config', required=True, help='Config JSON file where every image path used for inference must appear in the field TESTING ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--out-derivative', default='derivatives/label-SPINEPS', type=str, help='Derivative folder where the output data will be stored. (default="derivatives/label-SPINEPS")')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load variables
    config_path = os.path.abspath(args.config)
    derivatives_folder = args.out_derivative
    qc = True

    # Load config data
    # Read json file and create a dictionary
    with open(config_path, "r") as file:
        config_data = json.load(file)
    
    for di in config_data['TESTING']:
        path_image = os.path.join(config_data['DATASETS_PATH'], di['IMAGE'])
        # Create output path
        bids_path = path_image.split('sub-')[0]
        derivatives_path = os.path.join(bids_path, derivatives_folder)
        out_folder = os.path.join(derivatives_path, os.path.dirname(path_image.replace(bids_path,'')))
        if not os.path.exists(out_folder):
            print(f'{out_folder} was created')
            os.makedirs(out_folder)
        
        # Run SPINEPS prediction
        run_prediction(path_image, out_folder)

        # Generate QC
        if qc:
            spine_path = os.path.join(out_folder, os.path.basename(path_image).replace('.nii.gz', '_label-vert_dseg.nii.gz'))
            qc_path = os.path.join(derivatives_path, 'qc')
            subprocess.check_call([
                                "sct_qc", 
                                "-i", path_image,
                                "-s", spine_path,
                                "-d", spine_path,
                                "-p", "sct_deepseg_lesion",
                                "-plane", "sagittal",
                                "-qc", qc_path
                            ])

if __name__ == '__main__':
    main()