import numpy as np
import os
from glob import glob
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self,
                 text_dir: Union[str, Path],
                 motion_dir: Union[str, Path]):
        """
        Params
        -------
        text_dir : str, Path
            Directory with all the text descriptions files.
        motion_dir : str, Path
            Directory with all motion files.
        """

        # path of the text descripions and motions directory
        # .../text/
        # .../motions/
        self.text_dir = text_dir
        self.motion_dir = motion_dir

        # get sorted list of all text files and motion files
        self.text_files = sorted(glob(os.path.join(self.text_dir, '*.txt')))
        self.motion_files = sorted(glob(os.path.join(self.motion_dir, '*.npy')))
    
    def __len__(self):
        assert len(self.text_files) == len(self.motion_files), "Number of text files and motion files must be the same."
        return len(self.motion_files)
    
    def __getitem__(self, idx):
        # read npy motion file
        motion = np.load(self.motion_files[idx])

        # get the corresponding description for the associated motion
        text = []
        with open(self.text_files[idx]) as f:
            descriptions = f.readlines()
            for desc in descriptions:
                text.append(desc.split('#')[0].capitalize())

        return {
            "motion": motion,
            "text": text
        }