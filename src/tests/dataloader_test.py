 # Current system path
import os
from pathlib import Path

# Get the current working directory
working_path = Path(os.getcwd()).joinpath(Path("src"))
import sys
sys.path.append(str(working_path))


import unittest
import os
from data.nerp_datasets import MRIDataset, MRIDatasetUndersampling, MRIDatasetWithDistances
path = "data/knee_multicoil_train/file1000801.h5"
class TestAddFunction(unittest.TestCase):

    def test_MRIDataset(self):
        
        # Before the start of the test, make sure that file exsist
        assert os.path.exists(path), f"File is not exsist {path}"

        # Load this file
        dataset = MRIDataset(data_class="knee", transform=False, custom_file_or_path=path)

        # Get file name and compare with data
        self.assertEqual(dataset.file.name, "file1000801.h5")

        # Also try with normal operation
        dataset2 = MRIDataset(data_class="knee", transform=False)
        
        pass

    def test_MRIDatasetUndersampling_even_row_sampling(self):

        # Before the start of the test, make sure that file exsist
        assert os.path.exists(path), f"File is not exsist {path}"

        # Load this file
        dataset = MRIDataset(data_class="knee", transform=True, custom_file_or_path=path)

        # Load this file
        dataset_undersampled = MRIDatasetUndersampling(data_class="knee", transform=True, custom_file_or_path=path, undersampling="grid-5*5")

        self.assertEqual(len(dataset) , len(dataset_undersampled))
    
    def test_MRIDatasetUndersampling_nonetest(self):

        # Before the start of the test, make sure that file exsist
        assert os.path.exists(path), f"File is not exsist {path}"

        # Load this file
        dataset = MRIDatasetUndersampling(data_class="knee", transform=True, custom_file_or_path=path)

        # Load this file
        dataset_undersampled = MRIDatasetUndersampling(data_class="knee", transform=True, custom_file_or_path=path, undersampling="grid-5*5")

        self.assertEqual(len(dataset) , len(dataset_undersampled))
    
    def test_MRIDatasetWithDistances(self):

        # Before the start of the test, make sure that file exsist
        assert os.path.exists(path), f"File is not exsist {path}"

        # Load this file
        dataset = MRIDatasetWithDistances(data_class="knee", transform=False, custom_file_or_path=path)

        dataset_undersampled = MRIDatasetWithDistances(data_class="knee", transform=False, custom_file_or_path=path, undersampling="grid-5*5")

        self.assertEqual(len(dataset) , len(dataset_undersampled))

if __name__ == '__main__':
    unittest.main()