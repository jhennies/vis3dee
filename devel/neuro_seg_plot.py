
from mayavi import mlab
import numpy as np
import vigra
import sys


class NeuroSegPlot():

    def __init__(self):
        pass

    def path_in_segmentation(self):
        pass

    @staticmethod
    def start_figure():
        mlab.figure(bgcolor=(0, 0, 0))

    @staticmethod
    def add_path(path, anisotropy=[1, 1, 1]):
        # Plot the path -----------------------------------------------------------
        mlab.plot3d(
            path[0] * anisotropy[0],
            path[1] * anisotropy[1],
            path[2] * anisotropy[2],
            representation='wireframe',
            color=(1, 0, 0),
            line_width=5
        )

    @staticmethod
    def add_xyz_planes(image, anisotropy=[1, 1, 1], threshold=None, colormap='black-white'):
        # Raw data planes ---------------------------------------------------------
        src = mlab.pipeline.scalar_field(image)

        # Our data is not equally spaced in all directions:
        src.spacing = anisotropy
        src.update_image_data = True

        if threshold is not None:
            src = mlab.pipeline.threshold(src, **threshold)

        cut_plane = mlab.pipeline.scalar_cut_plane(src,
                                                   plane_orientation='x_axes',
                                                   colormap=colormap,
                                                   vmin=None,
                                                   vmax=None)
        cut_plane.implicit_plane.origin = (5, 0, 0)
        # cut_plane.implicit_plane.widget.enabled = False
        #
        cut_plane2 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='y_axes',
                                                    colormap=colormap,
                                                    vmin=None,
                                                    vmax=None)
        cut_plane2.implicit_plane.origin = (0, 2, 0)
        # cut_plane2.implicit_plane.widget.enabled = False

        cut_plane3 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='z_axes',
                                                    colormap=colormap,
                                                    vmin=None,
                                                    vmax=None)
        cut_plane3.implicit_plane.origin = (0, 0, 2)
        # cut_plane3.implicit_plane.widget.enabled = False

    @staticmethod
    def add_iso_surfaces(image, anisotropy=[1, 1, 1], colormap=None, color=None):

        for i in np.unique(image):

            if i == 0:
                continue

            # The surface of the selected segmentation object
            t_image = np.array(image)
            t_image[image != i] = 0
            obj = mlab.pipeline.scalar_field(t_image)
            t_image = None

            # Resize possibility no. two
            obj.spacing = anisotropy
            obj.update_image_data = True

            mlab.pipeline.iso_surface(
                obj, contours=[i],
                colormap=colormap,
                color=color,
                opacity=0.3,
                vmin=0, vmax=np.unique(image)[-1]
            )

    def path_in_segmentation_data_bg(self, raw_image, seg_image, path):
        """
        :param raw_image: np.ndarray data used for background
        :param seg_image: np.ndarray image segmentation used for iso-surface plot
        :param path: np.ndarray of the form:
            path = [[x_0, ..., x_n], [y_0, ..., y_n], [z_0, ..., z_n]]
        """

        from scipy.signal import medfilt

        mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

        # Plot the path -----------------------------------------------------------
        mlab.plot3d(
            path[0] * 10, path[1], path[2],
            representation='wireframe',
            color=(1, 0, 0),
            line_width=5
        )

        # Objects of the segmentation ----------------------------------------------
        print np.unique(seg_image)

        for i in np.unique(seg_image):

            if i == 0:
                continue

            # seg_image = vigra.filters.gaussianSmoothing(seg_image.astype(np.uint8), sigma=3)
            # seg_image = (seg_image > 0.4).astype(np.uint8)

            # # Resize possibility no. one
            # # Takes longer but like this I can smooth the surface of my segmentation
            # seg_image = lib.resize(seg_image, [10, 1, 1], mode='nearest')
            # seg_image = medfilt(seg_image, kernel_size=5)

            # The surface of the selected segmentation object
            t_seg_image = np.array(seg_image)
            t_seg_image[seg_image != i] = 0
            obj = mlab.pipeline.scalar_field(t_seg_image)
            t_seg_image = None

            # Resize possibility no. two
            obj.spacing = [10, 1, 1]
            obj.update_image_data = True

            mlab.pipeline.iso_surface(
                obj, contours=[i],
                colormap='Spectral',
                opacity=0.3,
                vmin=0, vmax=np.unique(seg_image)[-1]
            )

        # Raw data planes ---------------------------------------------------------
        src = mlab.pipeline.scalar_field(raw_image)
        # Our data is not equally spaced in all directions:
        src.spacing = [10, 1, 1]
        src.update_image_data = True

        cut_plane = mlab.pipeline.scalar_cut_plane(src,
                                                   plane_orientation='x_axes',
                                                   colormap='black-white',
                                                   vmin=None,
                                                   vmax=None)
        cut_plane.implicit_plane.origin = (5, 0, 0)
        # cut_plane.implicit_plane.widget.enabled = False
        #
        cut_plane2 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='y_axes',
                                                    colormap='black-white',
                                                    vmin=None,
                                                    vmax=None)
        cut_plane2.implicit_plane.origin = (0, 2, 0)
        # cut_plane2.implicit_plane.widget.enabled = False

        cut_plane3 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='z_axes',
                                                    colormap='black-white',
                                                    vmin=None,
                                                    vmax=None)
        cut_plane3.implicit_plane.origin = (0, 0, 2)
        # cut_plane3.implicit_plane.widget.enabled = False

        # mlab.show()
        self.show()

    @staticmethod
    def show():
        mlab.show()


if __name__ == '__main__':

    from hdf5_slim_processing import Hdf5Processing as hp
    import h5py

    # Parameters:
    anisotropy = [10, 1, 1]
    interpolation_mode = 'nearest'
    transparent = True
    opacity = 0.25
    # label = '118'
    label = '32'
    pathid = '1'
    surface_source = 'seg'

    # Specify the files
    raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    raw_file = 'cremi.splB.train.raw_neurons.crop.split_xyz.h5'
    raw_skey = ['z', '1', 'raw']

    seg_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170208_avoid_reduncant_path_calculation_sample_b_slice_z_train01_predict10_full/intermed/'
    seg_file = 'cremi.splB.train.segmlarge.crop.split_z.h5'
    seg_skey = ['z', '1', 'beta_0.5']

    gt_path = seg_path
    gt_file = 'cremi.splB.train.gtlarge.crop.split_z.h5'
    gt_skey = ['z', '1', 'neuron_ids']

    paths_path = seg_path
    paths_file = 'cremi.splB.train.paths.crop.split_z.h5'
    pathlist_file = 'cremi.splB.train.pathlist.crop.split_z.pkl'
    paths_skey = ['z_predict1', 'falsepaths', 'z', '1', 'beta_0.5']

    # # DEVELOP ----
    # # Specify the files
    # raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    # raw_file = 'cremi.splA.train.raw_neurons.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5'
    # raw_skey = ['x', '1', 'raw']
    #
    # seg_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    # seg_file = 'cremi.splA.train.segmlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # seg_skey = ['x', '1', 'beta_0.5']
    #
    # gt_path = seg_path
    # gt_file = 'cremi.splA.train.gtlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # gt_skey = ['x', '1', 'neuron_ids']
    #
    # paths_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    # paths_file = 'cremi.splA.train.paths.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # pathlist_file = 'cremi.splA.train.pathlist.crop.crop_x10_110_y200_712_z200_712.split_x.pkl'
    # paths_skey = ['predict', 'falsepaths', 'x', '1', 'beta_0.5']

    # crop = np.s_[0:10, 0:100, 0:100]

    # Load path
    paths = hp(filepath=paths_path + paths_file, nodata=True, skeys=[paths_skey])[paths_skey]
    print paths.keys()
    # label = paths.keys()[1]
    print 'Selected label = {}'.format(label)
    path = np.array(paths[label, pathid])

    import processing_lib as lib

    if surface_source == 'seg':
        # Load segmentation
        seg_image = np.array(hp(filepath=seg_path + seg_file, nodata=True, skeys=[seg_skey])[seg_skey])
        seg_image[seg_image != int(label)] = 0

    elif surface_source == 'gt':
        # Load ground truth
        seg_image = np.array(hp(filepath=gt_path + gt_file, nodata=True, skeys=[gt_skey])[gt_skey])
        gt_labels = np.unique(lib.getvaluesfromcoords(seg_image, path))
        t_seg_image = np.array(seg_image)
        for l in gt_labels:
            t_seg_image[t_seg_image == l] = 0
        seg_image[t_seg_image > 0] = 0
        t_seg_image = None

    else:
        sys.exit()

    crop = lib.find_bounding_rect(seg_image, s_=True)
    print 'crop = {}'.format(crop)
    seg_image = lib.crop_bounding_rect(seg_image, crop)

    path = np.swapaxes(path, 0, 1)
    path[0] = path[0] - crop[0].start
    path[1] = path[1] - crop[1].start
    path[2] = path[2] - crop[2].start

    # Load raw image
    raw_image = np.array(hp(filepath=raw_path + raw_file, nodata=True, skeys=[raw_skey])[raw_skey])

    raw_image = lib.crop_bounding_rect(raw_image, crop)

    plot = NeuroSegPlot()

    plot.path_in_segmentation_data_bg(raw_image, seg_image, path)

    # plot.show()