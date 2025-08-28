from typing import Union, Any, Literal, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_grid_utils import coordinate_grid
from torch_grid_utils.fftfreq_grid import dft_center
from torch_subpixel_crop import subpixel_crop_2d, subpixel_crop_3d

from .gaussians import Gaussian2D, Gaussian3D


def _refine_peaks_2d_torch(
    image: torch.Tensor,
    peak_data: torch.Tensor,
    boxsize: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
    sigma_bounds: tuple = (0.1, 10.0),
    amplitude_bounds: tuple = (0.01, 100.0),
    center_regularization: float = 0.0,
    amplitude_consistency: float = 0.0,
    sigma_consistency: float = 0.0,
) -> torch.Tensor:
    """
    Internal function to refine the positions of peaks in a 2D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 6) containing the fitted parameters for each peak.
        Each row contains [amplitude, y, x, sigma_x, sigma_y, loss].
    """
    
    # Crop regions around peaks
    boxes = subpixel_crop_2d(image, peak_data[...,1:3], boxsize).detach()
    # Prepare coordinates
    center = dft_center((boxsize, boxsize), rfft=False, fftshift=True)
    grid = coordinate_grid((boxsize, boxsize), center=center, device=image.device)

    # Initialize model
    model = Gaussian2D(amplitude=peak_data[..., 0],
                       center_x=torch.zeros_like(peak_data[..., 0]),
                       center_y=torch.zeros_like(peak_data[..., 0]),
                       sigma_x=peak_data[..., 3],
                       sigma_y=peak_data[..., 4],
                       sigma_bounds=sigma_bounds,
                       amplitude_bounds=amplitude_bounds).to(image.device)

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='none')  # Don't reduce to get per-peak losses

    # Fit the Gaussians
    per_peak_losses = None
    for _ in range(max_iterations):
        optimizer.zero_grad()

        # Calculate predicted values
        output = model(grid)
        # Calculate per-peak loss (mean over spatial dimensions, keep peak dimension)
        # output and boxes have shape: (num_peaks, height, width)
        loss_per_peak = torch.mean((output - boxes)**2, dim=[1, 2])  # Average over height, width
        mse_loss = torch.mean(loss_per_peak)  # MSE loss for backprop
        
        # Add regularization terms
        reg_loss = 0.0
        
        # L2 regularization for center shifts (keep centers close to 0)
        if center_regularization > 0:
            center_reg_loss = center_regularization * (torch.mean(model.center_x**2) + torch.mean(model.center_y**2))
            reg_loss += center_reg_loss
            
        # Amplitude consistency regularization (encourage similar amplitudes)
        if amplitude_consistency > 0:
            amp_var = torch.var(model.amplitude)
            amp_reg_loss = amplitude_consistency * amp_var
            reg_loss += amp_reg_loss
            
        # Sigma consistency regularization (encourage similar sigmas)
        if sigma_consistency > 0:
            sigma_x_var = torch.var(model.sigma_x)
            sigma_y_var = torch.var(model.sigma_y)
            sigma_reg_loss = sigma_consistency * (sigma_x_var + sigma_y_var)
            reg_loss += sigma_reg_loss
            
        loss = mse_loss + reg_loss
        
        # Store per-peak losses
        per_peak_losses = loss_per_peak.detach()
        
        # Check convergence
        if loss.item() < tolerance:
            break

        # Backpropagate and update
        loss.backward(retain_graph=False)  # Ensure no graph retention
        optimizer.step()

        # All parameter bounds now handled by log parameterization

    # Combine the (...,1) model parameters to a (...,6) tensor
    # and add the peak coordinates - keeping yx order
    fitted_params = torch.stack([
        model.amplitude,
        model.center_y + peak_data[..., 1],  # y coordinate first
        model.center_x + peak_data[..., 2],  # x coordinate second
        model.sigma_x,
        model.sigma_y,
        per_peak_losses  # per-peak MSE loss
    ], dim=-1)

    return fitted_params


def refine_peaks_2d(
    image: Any,
    peak_coords: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: Union[torch.Tensor, float] = 1.,
    sigma_x: Union[torch.Tensor, float] = 1.,
    sigma_y: Union[torch.Tensor, float] = 1.,
    sigma_bounds: tuple = (0.1, 10.0),
    amplitude_bounds: tuple = (0.01, 100.0),
    center_regularization: float = 0.0,
    amplitude_consistency: float = 0.0,
    sigma_consistency: float = 0.0,
    return_as: Literal["torch", "numpy", "dataframe"] = "torch",
) -> torch.Tensor:
    """
    Refine the positions of peaks in a 2D image by fitting 2D Gaussian functions.

    Parameters
    ----------
    image : Any
        A 2D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        containing the image data.
    peak_coords : torch.Tensor, np.ndarray, or pd.DataFrame
        A tensor-like object of shape (n, 2) containing the initial peak coordinates (y, x).
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for the optimization. Default is 1e-6.
    amplitude : Union[torch.Tensor, float], optional
        Initial amplitude of the Gaussian. Default is 1.0.
    sigma_x : Union[torch.Tensor, float], optional
        Initial standard deviation in the x direction. Default is 1.0.
    sigma_y : Union[torch.Tensor, float], optional
        Initial standard deviation in the y direction. Default is 1.0.
    sigma_bounds : tuple, optional
        Lower and upper bounds for sigma values (min, max). Default is (0.1, 10.0).
    amplitude_bounds : tuple, optional
        Lower and upper bounds for amplitude values (min, max). Default is (0.01, 100.0).
    center_regularization : float, optional
        L2 regularization strength for center shifts to keep them close to 0. Default is 0.0 (no regularization).
    amplitude_consistency : float, optional
        Regularization strength to encourage similar amplitudes across peaks. Default is 0.0 (no regularization).
    sigma_consistency : float, optional
        Regularization strength to encourage similar sigma values across peaks. Default is 0.0 (no regularization).

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 6) containing the fitted parameters for each peak.
        Each row contains [amplitude, y, x, sigma_x, sigma_y, loss].
    """
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if isinstance(peak_coords, pd.DataFrame):
        amplitude = torch.as_tensor(peak_coords["height"].to_numpy())
        peak_coords = torch.as_tensor(peak_coords[["y","x"]].to_numpy())
    if not isinstance(peak_coords, torch.Tensor):
        peak_coords = torch.as_tensor(peak_coords)

    num_peaks = peak_coords.shape[0]
    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.tensor([amplitude] * num_peaks, device=image.device)
    if not isinstance(sigma_x, torch.Tensor):
        sigma_x = torch.tensor([sigma_x] * num_peaks, device=image.device)
    if not isinstance(sigma_y, torch.Tensor):
        sigma_y = torch.tensor([sigma_y] * num_peaks, device=image.device)

    initial_peak_data = torch.stack([
        amplitude,
        peak_coords[:, 0],  # y
        peak_coords[:, 1],  # x
        sigma_x,
        sigma_y,
    ], dim=-1)

    refined_peak_data = _refine_peaks_2d_torch(
        image=image,
        peak_data=initial_peak_data,
        boxsize=boxsize,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        sigma_bounds=sigma_bounds,
        amplitude_bounds=amplitude_bounds,
        center_regularization=center_regularization,
        amplitude_consistency=amplitude_consistency,
        sigma_consistency=sigma_consistency,
    )

    if return_as=="torch":
        return refined_peak_data
    elif return_as=="numpy":
        return refined_peak_data.detach().cpu().numpy()
    elif return_as=="dataframe":
        return pd.DataFrame(refined_peak_data.detach().cpu().numpy(), columns=["amplitude", "y", "x", "sigma_x", "sigma_y", "loss"])
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")

def _refine_peaks_3d_torch(
    volume: torch.Tensor,
    peak_data: torch.Tensor,
    boxsize: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
    sigma_bounds: tuple = (0.1, 10.0),
    amplitude_bounds: tuple = (0.01, 100.0),
    center_regularization: float = 0.0,
    amplitude_consistency: float = 0.0,
    sigma_consistency: float = 0.0,
    background_consistency: float = 0.0,
    reshift_to_max: bool = True,
) -> torch.Tensor:
    """
    Internal function to refine the positions of peaks in a 3D tensor.

    Parameters
    ----------
    volume : torch.Tensor
        A 3D tensor containing the volume data.
    peak_data : torch.Tensor
        A tensor of shape (n, 8) containing the initial peak parameters.
        Each row contains [amplitude, z, y, x, sigma_x, sigma_y, sigma_z, background].
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int
        Maximum number of optimization iterations.
    learning_rate : float
        Learning rate for the optimizer.
    tolerance : float
        Convergence tolerance for the optimization.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 9) containing the refined parameters for each peak.
        Each row contains [amplitude, z, y, x, sigma_x, sigma_y, sigma_z, background, loss].
    """
    # Ensure boxsize is even
    if boxsize % 2 != 0:
        raise ValueError("boxsize must be even")

    # Crop regions around peaks
    boxes = subpixel_crop_3d(volume, peak_data[:, 1:4], boxsize).detach()
    if reshift_to_max:
        # Reshift the peak coordinates to the max within the cropped box
        # Find the index of the maximum value in each box
        flat_boxes = boxes.reshape(boxes.shape[0], -1)
        max_vals, max_idxs = torch.max(flat_boxes, dim=1)
        max_pos_z = (max_idxs // (boxsize * boxsize)) - (boxsize // 2)
        max_pos_y = ((max_idxs % (boxsize * boxsize)) // boxsize) - (boxsize // 2)
        max_pos_x = ((max_idxs % (boxsize * boxsize)) % boxsize) - (boxsize // 2)
        peak_data[:, 1] += max_pos_z.median()
        peak_data[:, 2] += max_pos_y.median()
        peak_data[:, 3] += max_pos_x.median()
        print("Updates to peak coordinates (z,y,x):", max_pos_z.median().item(), max_pos_y.median().item(), max_pos_x.median().item())
        # Re-crop with updated coordinates
        boxes = subpixel_crop_3d(volume, peak_data[:, 1:4], boxsize).detach()
    # Prepare coordinates
    center = dft_center((boxsize, boxsize, boxsize), rfft=False, fftshift=True)
    grid = coordinate_grid((boxsize, boxsize, boxsize), center=center, device=volume.device)

    # Initialize model
    model = Gaussian3D(
        amplitude=torch.zeros_like(peak_data[:, 0]) + torch.amax(boxes,dim=(1, 2, 3)),
        center_x=torch.zeros_like(peak_data[:, 0]),
        center_y=torch.zeros_like(peak_data[:, 0]),
        center_z=torch.zeros_like(peak_data[:, 0]),
        sigma_x=peak_data[:, 4],
        sigma_y=peak_data[:, 5],
        sigma_z=peak_data[:, 6],
        background=torch.zeros_like(peak_data[:, 0]) + torch.amin(boxes,dim=(1, 2, 3)),
        sigma_bounds=sigma_bounds,
        amplitude_bounds=amplitude_bounds
    ).to(volume.device)

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='none')  # Don't reduce to get per-peak losses

    # Fit the Gaussians
    per_peak_losses = None
    for _ in range(max_iterations):
        optimizer.zero_grad()

        # Calculate predicted values
        output = model(grid)
        # Calculate per-peak loss (mean over spatial dimensions, keep peak dimension)
        # output and boxes have shape: (num_peaks, depth, height, width)
        loss_per_peak = torch.mean((output - boxes)**2, dim=[1, 2, 3])  # Average over depth, height, width
        mse_loss = torch.mean(loss_per_peak)  # MSE loss for backprop
        
        # Add regularization terms
        reg_loss = 0.0
        
        # L2 regularization for center shifts (keep centers close to 0)
        if center_regularization > 0:
            center_reg_loss = center_regularization * (torch.mean(model.center_x**2) + torch.mean(model.center_y**2) + torch.mean(model.center_z**2))
            reg_loss += center_reg_loss
            
        # Amplitude consistency regularization (encourage similar amplitudes)
        if amplitude_consistency > 0:
            amp_var = torch.var(model.amplitude)
            amp_reg_loss = amplitude_consistency * amp_var
            reg_loss += amp_reg_loss
            
        # Sigma consistency regularization (encourage similar sigmas)
        if sigma_consistency > 0:
            sigma_x_var = torch.var(model.sigma_x)
            sigma_y_var = torch.var(model.sigma_y)
            sigma_z_var = torch.var(model.sigma_z)
            sigma_reg_loss = sigma_consistency * (sigma_x_var + sigma_y_var + sigma_z_var)
            reg_loss += sigma_reg_loss
            
        # Background consistency regularization (encourage similar backgrounds)
        if background_consistency > 0:
            bg_var = torch.var(model.background)
            bg_reg_loss = background_consistency * bg_var
            reg_loss += bg_reg_loss
            
        loss = mse_loss + reg_loss
        
        # Store per-peak losses
        per_peak_losses = loss_per_peak.detach()
        print(f"Iter {_}: {loss.item()} min: {per_peak_losses.min().item()} max: {per_peak_losses.max().item()}")
        # Check convergence
        if loss.item() < tolerance:
            break

        # Backpropagate and update
        loss.backward(retain_graph=False)  # Ensure no graph retention
        optimizer.step()

        # All parameter bounds now handled by log parameterization



    # Combine the (...,1) model parameters to a (...,9) tensor
    # and add the peak coordinates in zyx order
    fitted_params = torch.stack([
        model.amplitude,
        model.center_z + peak_data[:, 1],  # z coordinate first
        model.center_y + peak_data[:, 2],  # y coordinate second
        model.center_x + peak_data[:, 3],  # x coordinate third
        model.sigma_x,
        model.sigma_y,
        model.sigma_z,
        model.background,
        per_peak_losses  # per-peak MSE loss
    ], dim=-1)

    return fitted_params, boxes, output


def refine_peaks_3d(
    volume: Any,
    peak_coords: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: Union[torch.Tensor, float] = 1.,
    sigma_x: Union[torch.Tensor, float] = 1.,
    sigma_y: Union[torch.Tensor, float] = 1.,
    sigma_z: Union[torch.Tensor, float] = 1.,
    background: Union[torch.Tensor, float] = 0.0,
    sigma_bounds: tuple = (0.1, 10.0),
    amplitude_bounds: tuple = (0.01, 100.0),
    center_regularization: float = 0.0,
    amplitude_consistency: float = 0.0,
    sigma_consistency: float = 0.0,
    background_consistency: float = 0.0,
    return_as: Literal["torch", "numpy", "dataframe"] = "torch",
) -> torch.Tensor:
    """
    Refine the positions of peaks in a 3D volume by fitting 3D Gaussian functions.

    Parameters
    ----------
    volume : Any
        A 3D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        containing the volume data.
    peak_coords : torch.Tensor, np.ndarray, or pd.DataFrame
        A tensor-like object of shape (n, 3) containing the initial peak coordinates (z, y, x).
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for the optimization. Default is 1e-6.
    amplitude : Union[torch.Tensor, float], optional
        Initial amplitude of the Gaussian. Default is 1.0.
    sigma_x : Union[torch.Tensor, float], optional
        Initial standard deviation in the x direction. Default is 1.0.
    sigma_y : Union[torch.Tensor, float], optional
        Initial standard deviation in the y direction. Default is 1.0.
    sigma_z : Union[torch.Tensor, float], optional
        Initial standard deviation in the z direction. Default is 1.0.
    background : Union[torch.Tensor, float], optional
        Initial background value. Default is 0.0.
    sigma_bounds : tuple, optional
        Lower and upper bounds for sigma values (min, max). Default is (0.1, 10.0).
    amplitude_bounds : tuple, optional
        Lower and upper bounds for amplitude values (min, max). Default is (0.01, 100.0).
    center_regularization : float, optional
        L2 regularization strength for center shifts to keep them close to 0. Default is 0.0 (no regularization).
    amplitude_consistency : float, optional
        Regularization strength to encourage similar amplitudes across peaks. Default is 0.0 (no regularization).
    sigma_consistency : float, optional
        Regularization strength to encourage similar sigma values across peaks. Default is 0.0 (no regularization).
    background_consistency : float, optional
        Regularization strength to encourage similar background values across peaks. Default is 0.0 (no regularization).

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 9) containing the fitted parameters for each peak.
        Each row contains [amplitude, z, y, x, sigma_x, sigma_y, sigma_z, background, loss].
    """
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    if isinstance(peak_coords, pd.DataFrame):
        amplitude = torch.as_tensor(peak_coords["height"].to_numpy(),device=volume.device)
        peak_coords = torch.as_tensor(peak_coords[["z", "y", "x"]].to_numpy(),device=volume.device)
    if not isinstance(peak_coords, torch.Tensor):
        peak_coords = torch.as_tensor(peak_coords)

    num_peaks = peak_coords.shape[0]
    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.tensor([amplitude] * num_peaks, device=volume.device)
    if not isinstance(sigma_x, torch.Tensor):
        sigma_x = torch.tensor([sigma_x] * num_peaks, device=volume.device)
    if not isinstance(sigma_y, torch.Tensor):
        sigma_y = torch.tensor([sigma_y] * num_peaks, device=volume.device)
    if not isinstance(sigma_z, torch.Tensor):
        sigma_z = torch.tensor([sigma_z] * num_peaks, device=volume.device)
    if not isinstance(background, torch.Tensor):
        background = torch.tensor([background] * num_peaks, device=volume.device)

    initial_peak_data = torch.stack([
        amplitude,
        peak_coords[:, 0],  # z
        peak_coords[:, 1],  # y
        peak_coords[:, 2],  # x
        sigma_x,
        sigma_y,
        sigma_z,
        background
    ], dim=-1)

    refined_peak_data, boxes, output = _refine_peaks_3d_torch(
        volume=volume,
        peak_data=initial_peak_data,
        boxsize=boxsize,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        sigma_bounds=sigma_bounds,
        amplitude_bounds=amplitude_bounds,
        center_regularization=center_regularization,
        amplitude_consistency=amplitude_consistency,
        sigma_consistency=sigma_consistency,
        background_consistency=background_consistency,
    )

    if return_as == "torch":
        return refined_peak_data
    elif return_as == "numpy":
        return refined_peak_data.detach().cpu().numpy()
    elif return_as == "dataframe":
        return pd.DataFrame(refined_peak_data.detach().cpu().numpy(), columns=["amplitude", "z", "y", "x", "sigma_x", "sigma_y", "sigma_z", "background", "loss"])
    elif return_as == "diagnostic":
        # Return the boxes and output for diagnostic purposes
        return {
            "refined_peaks": pd.DataFrame(refined_peak_data.detach().cpu().numpy(), columns=["amplitude", "z", "y", "x", "sigma_x", "sigma_y", "sigma_z", "background", "loss"]),
            "boxes": boxes,
            "output": output
        }
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")
