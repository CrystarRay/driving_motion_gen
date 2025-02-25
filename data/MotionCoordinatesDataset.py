import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MotionCoordinatesDataset(Dataset):
    def __init__(self, data_root, partial_ratio=0.1, transform=None):
        """
        Args:
            data_root (str): Path to the main directory containing patient subfolders.
            partial_ratio (float): Fraction of the sequence to be used as the partial input.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        self.partial_ratio = partial_ratio
        self.transform = transform
        self.data_sequences = []

        merged_sequences = []
        patient_folders = sorted(os.listdir(data_root))

        for patient_folder in patient_folders:
            full_patient_path = os.path.join(data_root, patient_folder)
            if not os.path.isdir(full_patient_path):
                continue

            # Collect .npy or .npz files
            npy_files = sorted([os.path.join(full_patient_path, f) for f in os.listdir(full_patient_path) if f.endswith('.npy') or f.endswith('.npz')])

            if len(npy_files) != 2:
                print(f"Skipping {patient_folder}: Expected 2 .npy/.npz files but found {len(npy_files)}")
                continue

            def load_data(file_path):
                data = np.load(file_path)

                if isinstance(data, np.lib.npyio.NpzFile):
                    keys = list(data.keys())
                    print(f"Loading {file_path}, keys found: {keys}")
                    data = data[keys[0]]

                while len(data.shape) > 3 and data.shape[0] == 1:
                    data = np.squeeze(data, axis=0)  # Remove batch dimension

                return data


            data1 = load_data(npy_files[0])
            data2 = load_data(npy_files[1])

            # time dim check
            if data1.shape[1:] != data2.shape[1:]:
                print(f"Shape mismatch in {patient_folder}: {data1.shape} vs {data2.shape}")
                continue

            # Concatenate along T
            merged_sequence = np.concatenate([data1, data2], axis=0)  # (T1 + T2, 17, 3)
            merged_sequences.append(merged_sequence)

            print(f"{patient_folder}: Merged shape = {merged_sequence.shape}")

        # Find the global minimum sequence
        if not merged_sequences:
            raise ValueError("No valid data found after merging!")

        global_T_min = min(seq.shape[0] for seq in merged_sequences)
        print(f"Global minimum sequence length: {global_T_min}")

        # Step 3: Trim all sequences
        for seq in merged_sequences:
            trimmed_seq = seq[:global_T_min]
            self.data_sequences.append(torch.tensor(trimmed_seq, dtype=torch.float))

        print(f"Final dataset contains {len(self.data_sequences)} samples, each of length {global_T_min}")

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        full_seq = self.data_sequences[idx]

        # Create partial sequence
        T = full_seq.shape[0]
        partial_length = int(self.partial_ratio * T)
        partial_seq = full_seq[:partial_length]

        if partial_seq.shape[0] < T:
            pad_size = T - partial_seq.shape[0]
            pad_tensor = torch.zeros((pad_size, full_seq.shape[1], full_seq.shape[2]))
            partial_seq = torch.cat([partial_seq, pad_tensor], dim=0)

        return partial_seq, full_seq
