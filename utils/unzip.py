import os
import gzip
import shutil


def decompress_nii_gz(source_dir, destination_dir):
    failed = 0
    os.makedirs(destination_dir, exist_ok=True)
    item = 0
    for filename in os.listdir(source_dir):
        item += 1
        if item >= 700:
            break
        try:
            if filename.endswith('.nii.gz'):
                source_file = os.path.join(source_dir, filename)
                dest_filename = filename[:-3]
                destination_file = os.path.join(destination_dir, dest_filename)
                print(f"unzip: {source_file} -> {destination_file}")
                with gzip.open(source_file, 'rb') as f_in:
                    with open(destination_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            failed += 1
            continue
        
    
    print("scuccess!!!")
    print(f"failed: {failed}")

if __name__ == "__main__":
    source_directory = '/root/code/MRIclass/datasets/SLN_internal_mask/'
    destination_directory = '/root/code/MRIclass/datasets/mask_/' 
    decompress_nii_gz(source_directory, destination_directory)
    
