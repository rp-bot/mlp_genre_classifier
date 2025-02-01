import pandas as pd
import os
import numpy as np

dataset_dir = "data/PatchBanks"
output_csv_file = "data/PatchBanks/annotations.csv"

patch_banks_folders = [
    ("data/PatchBanks/edm_hse_id_001-004_wav", "house", 0),
    ("data/PatchBanks/edm_tr8_drm_id_001-0013_wav", "tr 808", 1),
    ("data/PatchBanks/edm_tr9_drm_id_001", "tr-909", 2),
    ("data/PatchBanks/hh_lfbb_lps_mid_001-009", "lofi-boom-bap", 3),
    ("data/PatchBanks/pop_rok_drm_id_001_wav", "pop rock", 4),
    ("data/PatchBanks/rtro_drm_id_001", "retro", 5),
    ("data/PatchBanks/wrld_lp_id_001", "latin percussion", 6),
    ("data/PatchBanks/wrld_smb_drm_8br_id_001_wav", "samba", 7)
]

metadata = []
for folder, class_label, class_id in patch_banks_folders:
    files = np.array(sorted(os.listdir(folder)))
    num_files = len(files)

    indices = np.linspace(0, num_files - 1, 1100, dtype=int)
    selected_files = files[indices]


    for file_name in selected_files:
        parts = file_name.replace(".wav", "").split("_")
        metadata.append({
            "file_path": os.path.join(folder, file_name),
            "class": class_label,
            "class_id": class_id
        })

if __name__ == '__main__':

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_csv_file)
