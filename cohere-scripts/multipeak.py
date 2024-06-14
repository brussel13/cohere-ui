# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script processes the multi-peak reconstructed image for visualization.
After the script is executed the experiment directory will contain image.vti file containing density, support, and the
three components of atomic displacement.
"""

__author__ = "Nick Porter"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['calc_geometry',
           'rotate_peaks',
           'twin_matrix',
           'center_mp',
           'write_vti',
           'process_dir']

from pathlib import Path
import numpy as np
from tvtk.api import tvtk
from multiprocessing import Process
from skimage import transform
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
import cohere_core.utilities as ut


def calc_geometry(instr_obj, shape, scan, o_twin):
    """Calculates the rotation matrix and voxel size for a given peak"""
    B_recip, _ = instr_obj.get_geometry(shape, scan, xtal=True)
    B_recip = np.stack([B_recip[1, :], B_recip[0, :], B_recip[2, :]])
    rs_voxel_size = np.max([np.linalg.norm(B_recip[:, i]) for i in range(3)])  # Units are inverse nanometers
    B_recip = o_twin @ B_recip
    return B_recip, rs_voxel_size


def rolloff3d(shape, sigma):
    mask = np.zeros(shape)
    mask[3*sigma:-3*sigma, 3*sigma:-3*sigma, 3*sigma:-3*sigma] = 1
    submask = np.zeros((2*sigma+1, 2*sigma+1, 2*sigma+1))
    a, b, c = np.mgrid[-sigma:sigma+1, -sigma:sigma+1, -sigma:sigma+1]
    submask[a**2+b**2+c**2 < sigma**2] = 1

    mask = ndi.binary_dilation(mask, submask).astype(float)
    mask = ndi.gaussian_filter(mask, sigma)
    return mask


def pad_to_cube(arr):
    padx, pady, padz = (np.max(arr.shape) - np.array(arr.shape)) // 2
    arr = np.pad(arr, ((padx, padx), (pady, pady), (padz, padz)))
    if len(np.unique(arr.shape)) != 1:
        padx, pady, padz = np.max(arr.shape) - np.array(arr.shape)
        arr = np.pad(arr, ((padx, 0), (pady, 0), (padz, 0)))
    return arr


def rotate_peaks(prep_obj, data, B_recip, voxel_size):
    """Rotates the diffraction pattern of a given peak to the common reference frame"""
    print("rotating diffraction pattern")
    vx_dims = np.array([np.linalg.norm(B_recip[:, i]) for i in range(3)])
    vx_dims = vx_dims / vx_dims.max()
    data = transform.rescale(data, 1/vx_dims, order=5)
    data = pad_to_cube(data)
    mask = np.ones_like(data)
    print(mask.shape)

    for i in range(3):
        B_recip[:, i] = B_recip[:, i] * vx_dims[i]

    matrix = voxel_size*np.linalg.inv(B_recip)
    center = np.array(data.shape) // 2
    translation = center - np.dot(matrix, center)
    data = ndi.affine_transform(data, matrix, order=5, offset=translation)
    mask = ndi.affine_transform(mask, matrix, order=1, offset=translation)
    mask[mask < 0.99] = 0

    final_size = prep_obj.final_size
    shp = np.array([final_size, final_size, final_size]) // 2

    # Pad the array to the largest dimensions
    shp1 = np.array(data.shape) // 2
    pad = shp - shp1
    pad[pad < 0] = 0
    data = np.pad(data, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])
    mask = np.pad(mask, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])

    # Crop the array to the final dimensions
    shp1 = np.array(data.shape) // 2
    start, end = shp1 - shp, shp1 + shp
    data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return data, mask.astype("?")


def refine_mask(init_mask, data):
    matrix = 0.8 * np.identity(3)
    center = np.array(data.shape) / 2
    offset = center - np.dot(matrix, center)
    mask = ndi.affine_transform(data, matrix, offset=offset, order=3)
    mask = ndi.gaussian_filter(mask, sigma=5) > 2

    dd = 5
    struct = np.zeros((dd, dd, dd))
    x, y, z = np.mgrid[-1:1:1j*dd, -1:1:1j*dd, -1:1:1j*dd]
    struct[x**2 + y**2 + z**2 < 1] = 1
    mask = ndi.binary_dilation(mask, structure=struct, iterations=1)
    return init_mask | np.invert(mask)


def twin_matrix(hklin, hklout, twin_plane, sample_axis):
    r1 = ut.normalize(hklin)
    r3 = ut.normalize(np.cross(hklin, hklout))
    r2 = ut.normalize(np.cross(r3, r1))
    rmat = np.stack([r1, r2, r3])

    twin_plane = rmat @ twin_plane
    theta = np.arccos(twin_plane @ sample_axis / (np.linalg.norm(twin_plane)*np.linalg.norm(sample_axis)))
    vec = ut.normalize(np.cross(twin_plane, sample_axis))

    return R.from_rotvec(vec * -theta).as_matrix()


def preprocess(preprocessor, instr_obj, scans_dirs, experiment_dir, mp_conf_map):
    try:
        o_twin = twin_matrix(mp_conf_map.get('hkl_in'),
                             mp_conf_map.get('hkl_out'),
                             mp_conf_map.get('twin_plane'),
                             mp_conf_map.get('sample_axis'))
    except KeyError:
        o_twin = np.identity(3)

    shape = instr_obj.get_scan_array(scans_dirs[0][0][1]).shape

    batches_rs_voxel_sizes = []
    batches_ds_voxel_sizes = []
    batches_B_recipes = []
    for batch in scans_dirs:
        first_scan = batch[0][0]
        B_recip, rs_voxel_size = calc_geometry(instr_obj, shape, first_scan, o_twin)
        batches_rs_voxel_sizes.append(rs_voxel_size)   # reciprocal-space voxel size in inverse nanometers
        batches_ds_voxel_sizes.append(2*np.pi/(rs_voxel_size*shape[0]))  # direct-space voxel size in nanometers
        batches_B_recipes.append(B_recip)

    rs_voxel_size = max(batches_rs_voxel_sizes)
    ds_voxel_size = min(batches_ds_voxel_sizes)
    # add the voxel size to config and save
    mp_conf_map["rs_voxel_size"] = rs_voxel_size
    mp_conf_map["ds_voxel_size"] = ds_voxel_size
    ut.write_config(mp_conf_map, ut.join(experiment_dir, 'conf', 'config_mp'))

    # run preprocessor for each batch (data set related to peak)
    processes = []
    conf_scans = mp_conf_map.get('scan').split(',')
    for i, batch in enumerate(scans_dirs):
        dirs = batch[0]
        scans = batch[1]
        geometry = {
            "peak_hkl": mp_conf_map.get('orientations')[i],
            "rmatrix": batches_B_recipes[i].tolist(),
            "lattice": mp_conf_map.get('lattice_size'),
            "rs_voxel_size": rs_voxel_size,
            "ds_voxel_size": ds_voxel_size,
            "final_size": mp_conf_map.get('final_size')
        }
        orientation = "".join(f"{o}" for o in geometry["peak_hkl"])
        save_dir = ut.join(experiment_dir, f'mp_{conf_scans[i]}_{orientation}')
        if not Path(save_dir).exists():
            Path(save_dir).mkdir()
        ut.write_config(geometry, ut.join(save_dir, 'geometry'))

        p = Process(target=preprocessor.process_batch,
                    args=(instr_obj.get_scan_array, batch, ut.join(save_dir, 'preprocessed_data', 'prep_data.tif'), experiment_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def center_mp(image, support):
    """
    Shifts the image and support arrays so the center of mass is in the center of array.
    Parameters
    ----------
    image, support : ndarray, ndarray
        image and support arrays to evaluate and shift
    Returns
    -------
    image, support : ndarray, ndarray
        shifted arrays
    """
    density = image[0]
    shape = density.shape
    max_coordinates = list(np.unravel_index(np.argmax(density), shape))
    for i in range(len(max_coordinates)):
        for j in range(10):
            image[j] = np.roll(image[j], int(shape[i] / 2) - max_coordinates[i], i)
        support = np.roll(support, int(shape[i] / 2) - max_coordinates[i], i)

    com = ndi.center_of_mass(density * support)

    # place center of mass in the center
    for i in range(len(shape)):
        for j, subimage in enumerate(image):
            image[j] = np.roll(subimage, int(shape[i] / 2 - com[i]), axis=i)
        support = np.roll(support, int(shape[i] / 2 - com[i]), axis=i)

    # set center displacement to zero, use as a reference
    half = np.array(shape) // 2
    for i in [1, 2, 3]:
        image[i] = image[i] - image[i, half[0], half[1], half[2]]

    return image, support


def write_vti(data, px, savedir, is_twin=False):
    # Create the vtk object for the data
    if is_twin:
        prepend = "twin_"
    else:
        prepend = ""
    print("Preparing VTK data")
    grid = tvtk.ImageData(dimensions=data[0].shape, spacing=(px, px, px))
    # Set the data to the image/support/distortion
    names = ["density", "u_x", "u_y", "u_z", "s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_zx", "support"]
    for img, name in zip(data, names):
        arr = tvtk.DoubleArray()
        arr.from_array(img.ravel())
        arr.name = name
        grid.point_data.add_array(arr)

    # print("Saving VTK")
    # Create the writer object
    writer = tvtk.XMLImageDataWriter(file_name=f"{savedir}/{prepend}full_data.vti")
    writer.set_input_data(grid)
    # Save the data
    writer.write()
    print(f"saved file: {savedir}/{prepend}full_data.vti")


def process_dir(exp_dir, rampups=1, make_twin=True):
    """
    Loads arrays from files in results directory. If reciprocal array exists, it will save reciprocal info in tif
    format.

    Parameters
    ----------
    exp_dir : str
        the directory where phasing results are saved
    rampups : int
        factor to apply to rampups operation, i.e. smoothing the image
    make_twin : bool
        if True visualize twin
    """
    res_dir = Path(exp_dir) / "results_phasing"
    save_dir = Path(exp_dir) / "results_viz"
    # create dir if does not exist
    print(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    for f in save_dir.iterdir():
        f.unlink()

    image = np.load(f"{res_dir}/reconstruction.npy")
    image = np.moveaxis(image, 3, 0)
    image[0] = image[0] / np.max(image[0])
    support = np.load(f"{res_dir}/support.npy")

    image, support = center_mp(image, support)
    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)
    np.save(f"{res_dir}/reconstruction.npy", np.moveaxis(image, 0, -1))

    px = ut.read_config(f"{exp_dir}/conf/config_mp")["ds_voxel_size"]

    write_vti(image, px, save_dir)

    if make_twin:
        image = np.flip(image, axis=(1, 2, 3))
        image[1:-1] *= -1
        if support is not None:
            support = np.flip(support)
            image, support = center_mp(image, support)
        if rampups > 1:
            image = ut.remove_ramp(image, ups=rampups)
        write_vti(image, px, save_dir, is_twin=True)
