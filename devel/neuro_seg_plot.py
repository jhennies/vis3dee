
from mayavi import mlab
import numpy as np
import vigra


class NeuroSegPlot():

    def __init__(self):
        pass

    def path_in_segmentation(self):
        pass

    def path_in_segmentation_data_bg(self, raw_image, seg_image):
        """
        :param raw_image: np.ndarray
        :return:
        """

        mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

        # Object of the segmentation ----------------------------------------------
        print np.unique(seg_image)

        # Extract the desired object
        seg_image[seg_image != 1] = 0
        # seg_image = vigra.filters.gaussianSmoothing(seg_image.astype(np.uint8), sigma=3)
        from scipy.signal import medfilt
        # seg_image = (seg_image > 0.4).astype(np.uint8)
        import processing_lib as lib
        seg_image = lib.resize(seg_image, [10, 1, 1], mode='nearest')
        seg_image = medfilt(seg_image, kernel_size=3)

        # The surface of the selected segmentation object
        # What is the contours argument?
        obj = mlab.pipeline.scalar_field(seg_image)
        # obj.spacing = [10, 1, 1]
        mlab.pipeline.iso_surface(
            obj, contours=2,
            color=(1, 0, 0),
            opacity=0.5
        )

        # Raw data planes ---------------------------------------------------------
        src = mlab.pipeline.scalar_field(raw_image)
        # Our data is not equally spaced in all directions:
        src.spacing = [11, 1, 1]
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

        mlab.show()
        # self.show()

    def show(self):
        mlab.show()


if __name__ == '__main__':

    from hdf5_slim_processing import Hdf5Processing as hp
    import h5py

    # Parameters:
    anisotropy = [10, 1, 1]
    interpolation_mode = 'nearest'
    transparent = True
    opacity = 0.25

    # Specify the files
    fpath = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170203_3d_plotting_develop/intermed/'
    segm_file = 'cremi.splA.train.gtlarge.crop.split_z.h5'
    segm_skeys = [['z', '1']]
    paths_file = 'cremi.splA.train.paths.crop.split_z.h5'
    pathlist_file = 'cremi.splA.train.pathlist.crop.split_z.pkl'
    paths_skeys = [['predict', '*', 'z', '1', 'beta_0.5']]

    raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    raw_file = 'cremi.splA.train.raw_neurons.crop.split_xyz.h5'
    raw_skey = ['z', '1', 'raw']

    seg_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170130_avoid_redundant_path_calculation_sample_a_slice_z_train_0_predict1_full/intermed/'
    seg_file = 'cremi.splA.train.segmlarge.crop.split_z.h5'
    seg_skey = ['z', '1', 'beta_0.5']

    crop = np.s_[0:10, 0:100, 0:100]

    # Load raw image
    raw_image = np.array(hp(filepath=raw_path + raw_file, nodata=True, skeys=[raw_skey])[raw_skey])
    import processing_lib as lib
    raw_image = lib.crop_bounding_rect(raw_image, crop)

    # Load segmentation
    seg_image = np.array(hp(filepath=seg_path + seg_file, nodata=True, skeys=[seg_skey])[seg_skey])
    seg_image = lib.crop_bounding_rect(seg_image, crop)

    # # Load image data
    # file_content = hp(filepath=fpath + segm_file, nodata=True, skeys=segm_skeys)
    # full_image = np.array(file_content['z', '1', 'neuron_ids'])
    # # Close the file
    # file_content.close()

    plot = NeuroSegPlot()

    plot.path_in_segmentation_data_bg(raw_image, seg_image)

    # plot.show()