
import numpy as np
from utils import load_dict

data_dir = "./data"

def get_E_H(xi, λi):
    """Get electric far field components for horizontally orientated dipole.

    Args:
        xi (int): index of the spatial displacement
        λi (int): index of the wavelength

    Returns:
        number of data points, E_x, E_y, E_z
    """
    result_E_zp_farfield = load_dict(f'{data_dir}/0deg/result_E_xp_farfield_n201_cartesian_x_{xi}.pkl')
    n = result_E_zp_farfield.shape[0]
    E_x, E_y, E_z = result_E_zp_farfield[:, :, 0, λi], result_E_zp_farfield[:, :, 1, λi], result_E_zp_farfield[:, :, 2, λi]
    return (n, E_y, E_z, E_x) # permuted because of Simon's rotation in the simulation file!

def get_E_V(xi, λi):
    """Get electric far field components for vertically orientated dipole.

    Args:
        xi (int): index of the spatial displacement
        λi (int): index of the wavelength

    Returns:
        number of data points, E_x, E_y, E_z
    """
    result_E_zp_farfield = load_dict(f'{data_dir}/90deg/result_E_xp_farfield_n201_cartesian_x_{xi}.pkl')
    n = result_E_zp_farfield.shape[0]
    E_x, E_y, E_z = result_E_zp_farfield[:, :, 0, λi], result_E_zp_farfield[:, :, 1, λi], result_E_zp_farfield[:, :, 2, λi]
    return (n, E_y, E_z, E_x) # permuted because of Simon's rotation in the simulation file!

n, E_H_x, E_H_y, E_H_z = get_E_H(1, 84)

np.savetxt(f"{data_dir}/0deg/E_H_x.csv", E_H_x, delimiter=",")
np.savetxt(f"{data_dir}/0deg/E_H_y.csv", E_H_y, delimiter=",")
np.savetxt(f"{data_dir}/0deg/E_H_z.csv", E_H_z, delimiter=",")

_, E_V_x, E_V_y, E_V_z = n, E_H_y.T, E_H_x.T, E_H_z.T
np.savetxt(f"{data_dir}/0deg/E_V_x_by_rotation.csv", E_V_x, delimiter=",")
np.savetxt(f"{data_dir}/0deg/E_V_y_by_rotation.csv", E_V_y, delimiter=",")
np.savetxt(f"{data_dir}/0deg/E_V_z_by_rotation.csv", E_V_z, delimiter=",")

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(np.abs(E_V_x)**2 + np.abs(E_V_y)**2 + np.abs(E_V_z)**2)
# axs[0].set_title('Vertical Dipole by Rotation')

_, E_V_x, E_V_y, E_V_z = get_E_V(1, 84)
np.savetxt(f"{data_dir}/90deg/E_V_x.csv", E_V_x, delimiter=",")
np.savetxt(f"{data_dir}/90deg/E_V_y.csv", E_V_y, delimiter=",")
np.savetxt(f"{data_dir}/90deg/E_V_z.csv", E_V_z, delimiter=",")

# axs[1].imshow(np.abs(E_V_x)**2 + np.abs(E_V_y)**2 + np.abs(E_V_z)**2)
# axs[1].set_title('Vertical Dipole by Simulation')
# plt.show()

pass