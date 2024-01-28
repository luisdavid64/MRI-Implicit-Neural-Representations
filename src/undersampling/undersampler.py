
from typing import Any, Tuple
import torch
import math
import numpy as np
from undersampling.utils import GOLDEN_RATIO, get_square_ordered_idxs, center_crop, verify_acc_factor
import matplotlib.pyplot as plt


# Define the supported methods
SUPORTED_UNDERSAMPLING_METHODS = ["grid", "random_line", "radial"]
class Undersampler():
    
    def __init__(self, undersamping_method : str) -> None:
        # Check if the supported method is provided
        assert undersamping_method in SUPORTED_UNDERSAMPLING_METHODS, f"Undersamping method: {undersamping_method} not supported"
        
        
        # Properties of this class
        self.undersampling_method = undersamping_method
        self.__mask_image = None
        self.__grid = None
        self.__grid_mask =None
        
    
    
    # Apply the undersampling
    # We need to implement the undersampling logic here instead of Dataset class
    # User of this class need to call this only after constructor call
    def apply(self, images_tensor: torch.Tensor, params : list) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()


        # Here apply the logic of choosing which undersampling method we are using it
        if self.undersampling_method == "grid":
            assert len(params) == 2, "Grid undersampling method's paramaters are not correct, it should have two parameters"
            # create mask, it will set __mask
            self.create_mask_for_grid_based_undersampling(H,W,params[0], params[1])
            
        elif self.undersampling_method =="random_line":
            assert len(params) == 1, "Random line undersampling method's paramaters are not correct, it should have one parameters"

            # create mask, it will set self.__mask
            self.create_mask_for_random_line_based_undersampling(H,W,params[0])

        elif self.undersampling_method == "radial":
            assert len(params) == 1, "Radial undersampling method's paramaters are not correct, it should have one parameters"

            # create mask, it will set self.__mask
            self.create_mask_for_radial_based_undersampling(images_tensor.shape,params[0])

        else:
            raise NotImplementedError()

        
        # Apply the mask
        # Maybe we do not need to apply the mask to image do we ? 
        masked_tensor = images_tensor * self.__mask_image.unsqueeze(0).unsqueeze(-1)

        # This will create grid as  self.__grid and it is undersampling mask as self.__grid_masked
        self.__create_grid(C,H,W)

        # masked_tensor, normal grid, 
        return masked_tensor, self.__grid, self.__grid_mask

    def __call__(self, images_tensor: torch.Tensor, params : list):
        self.apply(images_tensor, params)
    

    # Implement masks here, each function should set self.__mask according to the needs
        
    # GRID BASED UNDERSAMPLING
    # It is grid based undersampling not to confuse with grid construction
    def create_mask_for_grid_based_undersampling(self, image_h:int, image_w: int, grid_x: int = 3, grid_y: int = 3):
        mask = torch.zeros((image_h, image_w ), dtype=torch.bool) 

        # Apply the grid sampling mask
        mask[::grid_x, ::grid_y] = True

        # Set mask 
        self.__mask_image = mask


    # RANDOM LINE BASED UNDERSAMPLING
    def create_mask_for_random_line_based_undersampling(self, image_h:int, image_w: int, p: float):
        mask = torch.zeros((image_h, image_w ), dtype=torch.bool) 

        mask_x = torch.rand(image_h) <= p
        mask_y = torch.rand(image_w) <= p

        # Setting odd rows then collumns for the mask
        mask[mask_x,:] = True
        mask[:,mask_y] = True

        # Set mask 
        self.__mask_image = mask

    # RADIAL LINE BASED UNDERSAMPLING
    def create_mask_for_radial_based_undersampling(self,image_shape, acceleration: int):
        rng = np.random.RandomState()
        assert acceleration != 0, "Acceleration cannot be zero"

        max_dim = max(image_shape[1:3]) - max(image_shape[1:3]) % 2
        min_dim = min(image_shape[1:3]) - min(image_shape[1:3]) % 2
        num_nested_squares = max_dim // 2
        M = int(np.prod(image_shape[1:3]) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))
        mask = np.zeros((max_dim, max_dim), dtype=np.float32)

        t = rng.randint(low=0, high=1e4, size=1, dtype=int).item()

        for square_id in range(num_nested_squares):
            ordered_indices = get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )
            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((image_shape[1] % 2, 0), (image_shape[2] % 2, 0))

        mask = np.pad(mask, pad, constant_values=0)
        mask = center_crop(torch.from_numpy(mask.astype(bool)), image_shape[1:3])
        mask = ~mask

        # set mask
        self.__mask_image = mask


    # It creates grid for images not to confuse with grid undersampling
    def __create_grid(self, channel: int, image_h:int, image_w:int):
        assert self.__mask_image is not None, "Call apply() function first"

        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, channel),
                              torch.linspace(-1, 1, image_h),
                              torch.linspace(-1, 1, image_w))
        
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))


        # Create mask for each dimension and apply it from image mask
        Z_mask = torch.zeros(Z.shape, dtype=torch.bool)
        Z_mask[:,self.__mask_image] = True

        Y_mask = torch.zeros(Y.shape, dtype=torch.bool)
        Y_mask[:,self.__mask_image] = True

        X_mask = torch.zeros(X.shape, dtype=torch.bool)
        X_mask[:,self.__mask_image] = True


        grid_mask = torch.hstack((Z_mask.reshape(-1, 1),
                            Y_mask.reshape(-1, 1),
                            X_mask.reshape(-1, 1)))

        # Safety check
        assert grid.shape == grid_mask.shape, "Grid's shape and its mask shape should be same size"

        self.__grid = grid
        self.__grid_mask = grid_mask
    
    # Returns the grid constructed for image tensor and its mask
    def get_grid_and_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.__grid is not None or self.__grid_mask is not None, "Call apply() function first"
        
        return self.__grid, self.__grid_mask





## LEGACY SUPPORT, static functions
# We are not using them anymore

    @staticmethod
    def undersample_grid(images_tensor: torch.Tensor, grid_x: int = 3, grid_y: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()

        # Remove odd rows from the input tensor
        removed_odd_rows_tensor = images_tensor[:, ::grid_x, ::grid_y, :]

        # Create a coordinate grid based on the new dimensions
        new_H = removed_odd_rows_tensor.shape[1]
        new_W = removed_odd_rows_tensor.shape[2]

        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, C),
                                torch.linspace(-1, 1, new_H),
                                torch.linspace(-1, 1, new_W))

        # Reshape and stack the grids
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))

        

        return removed_odd_rows_tensor, grid
    
    def undersample_random_line(images_tensor: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()


        mask_x = torch.rand(H) < math.sqrt(p)
        mask_y = torch.rand(W) < math.sqrt(p)


        # Remove odd rows from the input tensor
        removed_odd_rows_tensor = images_tensor[:, mask_x,:, :]
        removed_odd_rows_tensor = removed_odd_rows_tensor[:, :, mask_y, :]

        # Apply same mask to linspace
        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, C),
                                torch.linspace(-1, 1, H)[mask_x],
                                torch.linspace(-1, 1, W)[mask_y])

        # Reshape and stack the grids
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))

        

        return removed_odd_rows_tensor, grid



    @staticmethod
    def undersample_radial(images_tensor: torch.Tensor, acceleration, save_mask=True) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState()
        assert images_tensor.dim() == 4, "For processing, please provide a 4-dimensional tensor as [batch_size, image_x, image_y, channel_n]"
        C,H,W,S = images_tensor.size()
        shape = images_tensor.shape
        max_dim = max(shape[1:3]) - max(shape[1:3]) % 2
        min_dim = min(shape[1:3]) - min(shape[1:3]) % 2
        num_nested_squares = max_dim // 2
        M = int(np.prod(shape[1:3]) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))
        mask = np.zeros((max_dim, max_dim), dtype=np.float32)
        t = rng.randint(low=0, high=1e4, size=1, dtype=int).item()

        for square_id in range(num_nested_squares):
            ordered_indices = get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )
            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[1] % 2, 0), (shape[2] % 2, 0))

        mask = np.pad(mask, pad, constant_values=0)
        mask = center_crop(torch.from_numpy(mask.astype(bool)), shape[1:3])
        if save_mask:
            plt.imshow(mask,cmap='gray')
            plt.savefig("undersampling_mask.png")
        Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, C),
                                torch.linspace(-1, 1, H),
                                torch.linspace(-1, 1, W))
        Z=Z[:,mask]
        Y=Y[:,mask]
        X=X[:,mask]
        # Reshape and stack the grids
        grid = torch.hstack((Z.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            X.reshape(-1, 1)))
        # removed_radial_tensor = images_tensor
        # removed_radial_tensor[:,mask,:] = 0
        removed_radial_tensor = images_tensor[:, mask, :] 
        print("Estimated Acceleration Factor: " + verify_acc_factor(mask))

        return removed_radial_tensor, grid



if __name__ == '__main__':
    # Example usage
    batch_size = 2
    x_dim = 8
    y_dim = 8
    channels = 2

    images_tensor = torch.rand((batch_size, x_dim, y_dim, channels))
    undersampeld_image, coordinate_grid = Undersampler.undersample_random_line(images_tensor, 0.5)
    #undersampeld_image, coordinate_grid = Undersampler.undersample_grid(images_tensor)
