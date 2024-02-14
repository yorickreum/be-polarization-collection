"""Utils used for fitting and loading/saving data.
"""


import pickle

import numpy as np

def save_dict(di_, filename_):
    """
    Save a dictionary to a .pkl file using pickle.

    Args:
        di_ (dict): The dictionary to be saved.
        filename_ (str): The name of the file to save the dictionary to.
    """
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    """
    Load a dictionary from a .pkl file using pickle.

    Parameters:
    filename_ (str): The path to the file containing the dictionary.

    Returns:
    dict: The loaded dictionary.
    """
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def gauss(x, A, μ, σ, B):
    """
    Calculate the value of a Gaussian function at a given point. Used for fitting.

    Parameters:
    - x: The input value.
    - A: Amplitude of the Gaussian.
    - μ: Mean of the Gaussian.
    - σ: Standard deviation of the Gaussian.
    - B: Constant offset.

    Returns:
    The value of the Gaussian function at the given point.
    """
    return A * np.exp(-(x-μ)**2 / (2. * σ**2)) + B

def lorentzian(λ, λ0, w, A, y0):
    """
    Calculate the Lorentzian function value at a given wavelength. Used for fitting.

    Parameters:
    λ (float): The wavelength at which to evaluate the Lorentzian function.
    λ0 (float): The center wavelength of the Lorentzian function.
    w (float): The full width at half maximum (FWHM) of the Lorentzian function.
    A (float): The amplitude of the Lorentzian function.
    y0 (float): The baseline value of the Lorentzian function.

    Returns:
    float: The value of the Lorentzian function at the given wavelength.
    """
    return y0 + (2*A/np.pi) * (w / (4 * (λ - λ0)**2 + w**2))

def lorentzian_H_to_A(H, w):
    """
    Converts the height H of a lorentzian curve with given FWHM wto the area under the curve.

    Parameters:
        H (float): Height
        w (float): FWHM

    Returns:
        float: Area under the curve.
    """
    return H * (np.pi * w) / 2

def lorentzian_A_to_H(A, w):
    """
    Converts the area A of a Lorentzian function with given FWHM w to its height H.

    Parameters:
        A (float): The area of the Lorentzian function.
        w (float): The FWHM the Lorentzian function.

    Returns:
        float: The height of the Lorentzian function.
    """
    return 2 * A / (np.pi * w)