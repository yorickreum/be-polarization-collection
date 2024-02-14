from datetime import datetime
import os

import pytz
from utils import load_dict

import numpy as np
from matplotlib import gridspec, pyplot as plt

# Wavelength index to be sliced from simulated farfield data
λi = 84  # yorick: 54 is 952.3 nm, 77 is 900.8 nm; simon: 84 is 930.28 nm

# input data and output plot directories
data_dir = "D:/Yorick/ma-thesis-lumerical-py/polarization-by-collection/data"
plot_dir = "D:/Yorick/ma-thesis-lumerical-py/polarization-by-collection/plots"


cmToIn = .3937
plt.rcParams.update({
    'savefig.pad_inches':  1,
    'figure.figsize': (16.5*cmToIn, 5*cmToIn),
    'figure.dpi': 1200,
    # 'figure.autolayout': True,
    # 'figure.constrained_layout.use': True,
    'font.size':  6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
    'xtick.major.width': .6,
    'ytick.major.width': .6,
    'legend.fancybox': False,
    'legend.fontsize': 6,
    'axes.titlesize': 7,
    'axes.labelpad': 2,
    'axes.linewidth': .6,
    'grid.linewidth': .6
    })

cmap = plt.colormaps['jet']
cmap.set_bad('black',1.)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    

# title_fontsize = 18
# label_fontsize = 18
# ticklabel_fontsize = 12


def get_E_H(xi, λi):
    # result_E_zp_farfield = load_dict(f'/home/ubuntu/lumerical-storage/simon_930_d260_h140_dipoley-sweep/eval/result_E_xp_farfield_n201_cartesian_pos_x_{xi}.pkl')
    result_E_zp_farfield = load_dict(f'{data_dir}/0deg/result_E_xp_farfield_n201_cartesian_x_{xi}.pkl')
    n = result_E_zp_farfield.shape[0]
    E_x, E_y, E_z = result_E_zp_farfield[:, :, 0, λi], result_E_zp_farfield[:, :, 1, λi], result_E_zp_farfield[:, :, 2, λi]
    return (n, E_y, E_z, E_x) # permuted because of Simon's rotation!

def get_E_V(xi, λi):
    # result_E_zp_farfield = load_dict(f'/home/ubuntu/lumerical-storage/simon_930_d260_h140_dipoley-sweep90deg/eval/result_E_xp_farfield_n201_cartesian_pos_x_{xi}.pkl')
    result_E_zp_farfield = load_dict(f'{data_dir}/90deg/result_E_xp_farfield_n201_cartesian_x_{xi}.pkl')
    n = result_E_zp_farfield.shape[0]
    E_x, E_y, E_z = result_E_zp_farfield[:, :, 0, λi], result_E_zp_farfield[:, :, 1, λi], result_E_zp_farfield[:, :, 2, λi]
    return (n, E_y, E_z, E_x) # permuted because of Simon's rotation!

def get_index(xi):
    result_index = load_dict(f'{data_dir}/0deg/result_index_x_{xi}.pkl')
    assert (result_index['index_y'] == result_index['index_z']).all()  # sanity check
    return result_index


def get_deltax(xi):
    # in nm
    # return (xi-1)*10 # Yorick
    return [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625][xi-1] # Simon


def integrate_dop(dop, angle):
    if angle == 0:
        return 0
    else: 
        circular_mask = np.zeros( (n,n) )
        for i in range(n):
            for j in range(n):
                circular_mask[i,j] = ((ux[i]**2 + uy[j]**2) >= 1) or ((180/np.pi) * np.arccos(np.sqrt( 1 - ux[i]**2 - uy[j]**2)) > angle)
        masked_dop = np.ma.masked_array(dop, mask=circular_mask)
        return masked_dop.sum()


def integrate_stokes_parameter(Si, angle):
    if angle == 0:
        return 0
    else: 
        circular_mask = np.zeros( (n,n) )
        for i in range(n):
            for j in range(n):
                circular_mask[i,j] = ((ux[i]**2 + uy[j]**2) >= 1) or ((180/np.pi) * np.arccos(np.sqrt( 1 - ux[i]**2 - uy[j]**2)) > angle)
        masked_Si = np.ma.masked_array(Si, mask=circular_mask)
        return masked_Si.sum()
    
def integrate_stokes(S1, S2, S3, angle):
    if angle == 0:
        return 0
    else: 
        circular_mask = np.zeros( (n,n) )
        for i in range(n):
            for j in range(n):
                circular_mask[i,j] = ((ux[i]**2 + uy[j]**2) >= 1) or ((180/np.pi) * np.arccos(np.sqrt( 1 - ux[i]**2 - uy[j]**2)) > angle)
        masked_S1 = np.ma.masked_array(S1, mask=circular_mask)
        masked_S2 = np.ma.masked_array(S2, mask=circular_mask)
        masked_S3 = np.ma.masked_array(S3, mask=circular_mask)
        return masked_S1.sum() + masked_S2.sum() + masked_S3.sum()
    

for xi in [1]: # xi is index for dipole displacement sweep in the x-direction, 1 is no displacement

    is_a = (xi == 1)

    fig = plt.figure()

    n, E_H_x, E_H_y, E_H_z = get_E_H(xi, λi)
    _, E_V_x, E_V_y, E_V_z = get_E_V(xi, λi)

    ux, uy = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
    rs = [ [ (180/np.pi) * np.arccos(np.sqrt( 1 - ux[i]**2 - uy[j]**2)) if (ux[i]**2 + uy[j]**2) < 1 else 90 for j in range(n)] for i in range(n)]
    phis = [ [np.arctan2(ux[i], uy[j]) for j in range(n)] for i in range(n)] # @TODO formular is uy/ux, but arctan2 inverts arg order, so it's like that?

    circular_mask = np.zeros( (n,n) )
    for i in range(n):
        for j in range(n):
            circular_mask[i,j] = (np.sqrt(ux[i]**2 + uy[j]**2) > 1)

    I = (np.abs(E_H_x)**2 + np.abs(E_V_x)**2) + (np.abs(E_H_y)**2 + np.abs(E_V_y)**2)
    # I = (np.abs(E_H_x)**2) + (np.abs(E_H_y)**2)
    Q = (np.abs(E_H_x)**2 + np.abs(E_V_x)**2) - (np.abs(E_H_y)**2 + np.abs(E_V_y)**2)
    # Q = (np.abs(E_H_x)**2 - np.abs(E_H_y)**2 )
    U =  2 * np.real( E_H_x * np.conjugate(E_H_y) + E_V_x * np.conjugate(E_V_y) )
    # U =  2 * np.real( E_H_x * np.conjugate(E_H_y))
    V = -2 * np.imag( E_H_x * np.conjugate(E_H_y) + E_V_x * np.conjugate(E_V_y) )
    # V = -2 * np.imag( E_H_x * np.conjugate(E_H_y) )

    dop = np.sqrt(Q**2 + U**2 + V**2)
    dolp = np.sqrt(Q**2 + U**2)
    
    # Imax = np.max(I)
    Qnorm = Q / I
    Unorm = U / I
    Vnorm = V / I
    dopNorm = np.sqrt(Qnorm**2 + Unorm**2 + Vnorm**2)
    dolpNorm = np.sqrt(Qnorm**2 + Unorm**2)

    Iint = np.nansum(I)
    Qint = np.nansum(Q)
    Uint = np.nansum(U)
    Vint = np.nansum(V)

    title_pos = 1.05

    collection_angle = 90 # 54.1 for Giora's NA

    IintNA = integrate_stokes_parameter(I, collection_angle)
    QintNA = integrate_stokes_parameter(Q, collection_angle)
    UintNA = integrate_stokes_parameter(U, collection_angle)
    VintNA = integrate_stokes_parameter(V, collection_angle)

    Iavg = np.nanmean(I)
    Qavg = np.nanmean(Q)
    Uavg = np.nanmean(U)
    Vavg = np.nanmean(V)

    Imax = np.nanmax(I)
    Qmax = np.nanmax(Q)
    Umax = np.nanmax(U)
    Vmax = np.nanmax(V)

    cols = 5
    rows = 2

    # gs_outer = fig.add_gridspec(rows, 1, height_ratios=[1, 1], hspace=.1, wspace=0, left=.02, right=.98)
    gs_outer = fig.add_gridspec(1, 1, height_ratios=[1], hspace=.1, left=.06, right=.98)

    for rowi in [1]: # range(2):
        # axIndex = fig.add_subplot((rowi*cols)+2)

        gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = gs_outer[0], width_ratios=[1,1,3], wspace=.15)


        gsIndex = gs[:,0].subgridspec(2, 1, height_ratios=[19,1])
        gsIntensity = gs[:,1].subgridspec(2, 1, height_ratios=[19,1])
        gsStokes = gs[:,2].subgridspec(2, 3, wspace=0, height_ratios=[19,1])

        # gs_intensity = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[0:2,1], height_ratios=[19,1], hspace=-.1)
        # gs_stokes = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=gs[0:2,2:5], height_ratios=[19,1], hspace=-.2, wspace=0)

        axIndex = fig.add_subplot(gsIndex[0, 0])
        axIndexText = fig.add_subplot(gsIndex[1,0])
        
        axStokesIcart = fig.add_subplot(gsIntensity[0, 0])
        axStokesI = fig.add_subplot(gsIntensity[0, 0], polar="True", frameon=True)
        # axStokesI = fig.add_axes(axStokesIcart.get_position(), polar="True", frameon=True)
        # axStokesIcart = fig.add_axes(axStokesI.get_position().bounds)
        axStokesI.grid(False)
        axStokesIcbar = fig.add_subplot(gsIntensity[1, 0])

        axStokesQcart = fig.add_subplot(gsStokes[0, 0])
        axStokesQ = fig.add_subplot(gsStokes[0, 0], polar="True", frameon=True)
        axStokesQ.grid(False)
        # axStokesQcbar = fig.add_subplot(gs_stokes[1, 0])
        # axStokesQcbar.set_axis_off()

        axStokesUcart = fig.add_subplot(gsStokes[0, 1], sharey=axStokesQcart)
        axStokesU = fig.add_subplot(gsStokes[0, 1], polar="True", frameon=True)
        axStokesU.grid(False)
        axStokesUcbar = fig.add_subplot(gsStokes[1, 0:4])

        axStokesVcart = fig.add_subplot(gsStokes[0, 2], sharey=axStokesQcart)
        axStokesV = fig.add_subplot(gsStokes[0, 2], polar="True", frameon=True)
        axStokesV.grid(False)
        # axStokesVcbar = fig.add_subplot(gs_stokes[1, 2])
        # axStokesVcbar.set_axis_off()

        showDop = False
        if showDop: # Dop
            axDopCart = fig.add_subplot(gs[0, 5])
            axDop = fig.add_subplot(gs[0, 5], polar="True")
            axDop.grid(False)
            axDopcbar = fig.add_subplot(gs[1, 5])

            axDolpcart = fig.add_subplot(gs[0, 6])
            axDolp = fig.add_subplot(gs[0, 6], polar="True")
            axDolp.grid(False)
            axDolpcbar = fig.add_subplot(gs[1, 6])

            axDopIntegrated = fig.add_subplot(gs[0, 7])
            axDopIntegrated.grid(False)

        rmax = collection_angle

        if rowi == 0:
            axIndex.axis('off')
        else:
            result_index = get_index(xi)
            x_min_index, x_max_index, y_min_index, y_max_index = result_index['y'].min(), result_index['y'].max(), result_index['z'].min(), result_index['z'].max()
            im_index = axIndex.imshow(
                np.real(result_index['index_y'][0, :, :, 0]).T,
                origin="lower",
                interpolation='none',
                extent=[x_min_index, x_max_index, y_min_index, y_max_index],
                # extent=[-index_extent,+index_extent,-index_extent,+index_extent],
                aspect=1,
                alpha=.1,
                cmap='Greys'
            )
            x_ticks = np.arange(-2600e-9, 2501e-9, 400e-9)
            x_ticks = np.arange(-2600e-9, 2501e-9, 400e-9)
            # axStokesIOrIndex.axhline(0, c='lightgray', linestyle='dashed',linewidth=1, alpha=.5)
            # axStokesIOrIndex.axvline(0, c='lightgray', linestyle='dashed',linewidth=1, alpha=.5)
            axIndex.grid(True, alpha=.5, zorder=0)
            axIndex.set_xticks( ticks=x_ticks, labels=[ f'{(tick*1e9):.0f}' for tick in x_ticks] )
            axIndex.set_yticks( ticks=x_ticks, labels=[ f'{(tick*1e9):.0f}' for tick in x_ticks] )
            index_extent = 800e-9
            axIndex.set_xlim( (-index_extent, index_extent) )
            axIndex.set_ylim( (-index_extent, index_extent) )
            axIndex.set_xlabel("x (nm)")
            axIndex.set_ylabel("y (nm)")
            axIndex.set_title('Position of the Dipoles', y=title_pos, color=('black' if is_a else 'white'))
            axIndex.tick_params(direction='in', top=True, right=True)
            delta_x = get_deltax(xi) * 1e-9
            axIndex.arrow(-100e-9 + delta_x, 0, 200e-9, 0,
                width=20e-9, head_length=40e-9,
                shape='full', color='r', zorder=1)
            axIndex.arrow(+100e-9 + delta_x, 0, -200e-9, 0,
                width=20e-9, head_length=40e-9,
                shape='full', color='r', zorder=1)
            axIndex.arrow(0 + delta_x, -100e-9, 0, 200e-9,
                width=20e-9, head_length=40e-9,
                shape='full', color='b', zorder=2)
            axIndex.arrow(0 + delta_x, +100e-9, 0, -200e-9,
                width=20e-9, head_length=40e-9,
                shape='full', color='b', zorder=2)
            axIndex.scatter(delta_x, 0, s=5, c='yellow', marker='o', zorder=3)
            axIndexText.set_axis_off()
            axIndexText.text(.5, -.5, f"{(delta_x*1e9):.0f} nm displacement", horizontalalignment='center', verticalalignment='center', transform=axIndexText.transAxes, weight='bold')

        # Intensity
        if rowi == 0:
            # np.abs(result_total_E_zp_farfield[i,j,0, λi_init])**2 + np.abs(result_total_E_zp_farfield[i,j,1, λi_init])**2 + np.abs(result_total_E_zp_farfield[i,j,2, λi_init])**2
            imStokesI = axStokesI.pcolormesh(phis,rs,
                np.ma.masked_array(I.T, mask=circular_mask),
                cmap=cmap, rasterized=True, shading='gouraud')
            axStokesI.set_title(f'$I = |E_{{H,x}}|^2 + |E_{{H,y}}|^2 + |E_{{V,x}}|^2 + |E_{{V,y}}|^2$ \n int: ${Iint:2.3}$, max: ${Imax:2.3}$', y=1.1)
        else:
             # np.abs(result_total_E_zp_farfield[i,j,0, λi_init])**2 + np.abs(result_total_E_zp_farfield[i,j,1, λi_init])**2 + np.abs(result_total_E_zp_farfield[i,j,2, λi_init])**2
            imStokesI = axStokesI.pcolormesh(phis,rs,
                np.ma.masked_array((I/Imax).T, mask=circular_mask),
                cmap=cmap, rasterized=True, shading='gouraud')
            axStokesI.set_title(f'Normalized Intensity', y=title_pos, color=('black' if is_a else 'white'))
            imStokesI.set_clim(vmin=0, vmax=1)
        fig.colorbar(imStokesI, cax=axStokesIcbar, orientation="horizontal", panchor=(0.5, 0))

        # S1
        if rowi == 0:
            imStokesQ = axStokesQ.pcolormesh(phis,rs,
                    np.ma.masked_array(Q.T, mask=circular_mask),
                    cmap=cmap, rasterized=True, shading='gouraud')
            axStokesQ.set_title(f'$Q = |E_{{H,x}}|^2 - |E_{{H,y}}|^2 + |E_{{V,x}}|^2 - |E_{{V,y}}|^2$  \n int: ${Qint:2.3}$, max: ${Qmax:2.3}$', y=1.1)
        elif rowi == 1:
            imStokesQ = axStokesQ.pcolormesh(phis,rs,
                    np.ma.masked_array(Qnorm.T, mask=circular_mask),
                    cmap=cmap, rasterized=True, shading='gouraud')
            imStokesQ.set_clim(vmin=-1, vmax=1)
            # axStokesQ.set_title(f'$Q_\mathrm{{norm}} = Q / I$ \n int: ${Qint:2.3}$, \n avg: ${Qavg:2.3}$, \n max: ${Qmax:2.3}$', y=1.1)
            axStokesQ.set_title(f'Integrated $S_1 = {(round(QintNA/IintNA, 2) + 0.):.2f}$ ', y=title_pos)
        # fig.colorbar(imStokesQ, cax=axStokesQcbar, orientation="horizontal", panchor=(0.5, 0), pad=.5)

        if rowi == 0:
            imStokesU = axStokesU.pcolormesh(phis, rs,
                                            np.ma.masked_array(U.T, mask=circular_mask),
                                            cmap=cmap, rasterized=True, shading='gouraud')
            axStokesU.set_title(f'$U = 2 \mathrm{{Re}} \left( E_{{H,x}} E_{{H,y}}^* + E_{{V,x}} E_{{V,y}}^*  \\right)$ \n int: ${Uint:2.3}$, max: ${Umax:2.3}$', y=1.1)
        elif rowi == 1:
            imStokesU = axStokesU.pcolormesh(phis, rs,
                                            np.ma.masked_array(Unorm.T, mask=circular_mask),
                                            cmap=cmap, rasterized=True, shading='gouraud')
            imStokesU.set_clim(vmin=-1, vmax=1)
            axStokesU.set_title(f'$S_2 = {(round(UintNA/IintNA, 2) + 0.):.2f}$ ', y=title_pos)
            # axStokesU.set_title(f'$U_\mathrm{{norm}} = U / I$ \n int: ${Uint:2.3}$, \n avg: ${Uavg:2.3}$, \n max: ${Umax:2.3}$', y=1.1)
        fig.colorbar(imStokesU, cax=axStokesUcbar, orientation="horizontal", panchor=(0.5, 0))

        if rowi == 0:
            imStokesV = axStokesV.pcolormesh(phis, rs,
                                            np.ma.masked_array(V.T, mask=circular_mask),
                                            cmap=cmap, rasterized=True, shading='gouraud')
            axStokesV.set_title(f'$V = -2 \mathrm{{Im}} \left( E_{{H,x}} E_{{H,y}}^* + E_{{V,x}} E_{{V,y}}^*  \\right)$ \n int: ${Vint:2.3}$, max: ${Vmax:2.3}$', y=1.1)
        elif rowi == 1:
            imStokesV = axStokesV.pcolormesh(phis, rs,
                                            np.ma.masked_array(Vnorm.T, mask=circular_mask),
                                            cmap=cmap, rasterized=True, shading='gouraud')
            imStokesV.set_clim(vmin=-1, vmax=1)
            axStokesV.set_title(f'$S_3 = {(round(VintNA/IintNA, 2) + 0.):.2f}$ ', y=title_pos)
        # fig.colorbar(imStokesV, cax=axStokesVcbar, orientation="horizontal", panchor=(0.5, 0))

        if showDop:
            if rowi == 0:
                imDop = axDop.pcolormesh(phis, rs,
                                                np.ma.masked_array(dop.T, mask=circular_mask),
                                                cmap=cmap, rasterized=True, shading='gouraud')
                axDop.set_title(
                    'Degree of polarization \n $p = \sqrt{Q^2 + U^2 + V^2}$', y=1.1)
            if rowi == 1:
                imDop = axDop.pcolormesh(phis, rs,
                                                np.ma.masked_array(dopNorm.T, mask=circular_mask),
                                                cmap=cmap, rasterized=True, shading='gouraud')
                axDop.set_title('$p_\mathrm{norm} = \sqrt{Q_\mathrm{norm}^2 + U_\mathrm{norm}^2 + V_\mathrm{norm}^2}$', y=1.1)
            fig.colorbar(imDop, cax=axDopcbar, orientation="horizontal", panchor=(0.5, 0))

            if rowi == 0:
                imDolp = axDolp.pcolormesh(phis, rs,
                                                np.ma.masked_array(dolp.T, mask=circular_mask),
                                                cmap=cmap, rasterized=True, shading='gouraud')
                axDolp.set_title(
                    'Degree of linear polarization \n $p_L = \sqrt{Q^2 + U^2}$', y=1.1)
            if rowi == 1:
                imDolp = axDolp.pcolormesh(phis, rs,
                                                np.ma.masked_array(dolpNorm.T, mask=circular_mask),
                                                cmap=cmap, rasterized=True, shading='gouraud')
                axDolp.set_title('$p_{L, \mathrm{norm}} = \sqrt{Q_\mathrm{norm}^2 + U_\mathrm{norm}^2}$', y=1.1)
            fig.colorbar(imDolp, cax=axDolpcbar, orientation="horizontal", panchor=(0.5, 0))

        axsPol = [axStokesI, axStokesQ, axStokesU, axStokesV]
        if showDop:
            axsPol.append( [axDop, axDolp] )
        # if rowi == 0:
        #    axsPol.append(axIndex)
        for axPol in axsPol:
            axPol.set_facecolor("gray")
            rgrids = [15,30,45]
            axPol.set_rgrids( rgrids, labels=["" for s in rgrids], color='white', position='12' ) # labels=[f'{s} °' for s in rgrids], 
            axPol.set_rmax(rmax)
            thetagrid = [0, 45, 90, 135, 180, 225, 270, 315]
            axPol.set_thetagrids(thetagrid, labels=["" for s in thetagrid])
            axPol.grid(color='white', ls='dashed')

        axsPolCar = [axStokesIcart, axStokesQcart, axStokesUcart, axStokesVcart]
        if showDop:
            axsPolCar.append( [axDopCart, axDolpcart] )
        for axPolCart in axsPolCar:
            # ticks = [-60, -45, -15, 15, 45, 60]
            ticks = [-90, -45, 0, 45, 90]
            axPolCart.set_xticks( ticks, labels=[f"{s}°" for s in ticks] )
            axPolCart.set_yticks( ticks, labels=[f"{s}°" for s in ticks] )
            axPolCart.set_aspect('equal')
            axPolCart.set_xlim( -rmax, rmax )
            axPolCart.set_ylim( -rmax, rmax )
            axPolCart.tick_params(direction='in', top=True, bottom=True)
            axPolCart.tick_params(direction='inout', left=True, right=True)
        axStokesIcart.tick_params(direction='in', left=True)
        axStokesQcart.tick_params(direction='inout', left=True)
        axStokesQcart.tick_params(direction='in', right=True)
        axsPolCarNoYLabels = [axStokesUcart, axStokesVcart]
        for axPolCart in axsPolCarNoYLabels:
            axPolCart.set_yticklabels([])
            axPolCart.tick_params(direction='in', top=True, right=True)


        axsCbar = [axStokesIcbar, axStokesUcbar] # axStokesQcbar, axStokesVcbar
        if showDop:
            axsCbar.append( [axDopcbar, axDolpcbar] )
        for cbarAx in axsCbar:
            cbarAx.tick_params(direction='inout')
            cbarAx.set_box_aspect(.055)
            cbarAx.set_xticks([-1,-.5,0,.5,1])
            cbarAx.autoscale()

        if showDop:
            angles = range(0, 91, 1)
            inetgratedStokes = [integrate_stokes(Q, U, V, a) for a in angles]
            ticks = np.arange(0, 90, 10)
            axDopIntegrated.set_xticks( ticks=ticks, labels=[ f'{(tick):.0f}' for tick in ticks] )
            axDopIntegrated.set_xlabel( "polar angle $\; \\theta_0$ [°]" )
            axDopIntegrated.set_aspect( 1.35 * (max(angles)-min(angles)) / (max(inetgratedStokes)-min(inetgratedStokes)) )
            imDopIntegrated = axDopIntegrated.plot(angles, inetgratedStokes)
            axDopIntegrated.set_title('$\int_{{0}}^{\\theta_0} Q + U + V \mathrm{d} \\theta$', y=1.05)
            if rowi == 1:
                axDopIntegrated.set_title('$\int_{{0}}^{\\theta_0} Q_\mathrm{norm} + U_\mathrm{norm} + V_\mathrm{norm} \mathrm{d} \\theta$', y=1.05)

    lambda_selected = 930e-9
    fig.suptitle(f'Dipole Displacement $\Delta x = {get_deltax(xi)}$ nm at {(lambda_selected* 1e9):.0f} nm',)

    tz = pytz.timezone('Europe/Berlin')
    berlin_now = datetime.now(tz)
    now_string = berlin_now.strftime("%d.%m.%Y %H:%M:%S")

    # plt.subplot_tool()
    # plt.show()

    fig.text(.01,.90, "(a)" if is_a else "(b)")

    filename = f'farfield_stokes_x_pos_{xi:02d}.png'
    plt.savefig(f'{plot_dir}/{filename}')
    plt.close()
    print(f"Just wrote \"{filename}\".")
