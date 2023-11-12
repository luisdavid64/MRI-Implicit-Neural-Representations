# Current system path
import os
from pathlib import Path

# Get the current working directory
working_path = Path(os.getcwd()).joinpath(Path("src"))
import sys
sys.path.append(str(working_path))


import unittest
import os
from data.nerp_datasets import MRIDataset

class TestAddFunction(unittest.TestCase):

    def test_MRIDataset(self):
        
        path = "data/knee_multicoil_train/file1000801.h5"

        dataset = MRIDataset(data_class="knee", transform=False, custom_file_or_path=path)
        dataset2 = MRIDataset(data_class="knee", transform=False)
        pass

if __name__ == '__main__':
    unittest.main()