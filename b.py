#!/usr/bin/env python
import multiprocessing
import vtk

from invesalius.data import cy_mesh
import invesalius.data.imagedata_utils as image_utils
from invesalius.data import mask
import invesalius.data.slice_ as sl
from invesalius.data import surface_process
import invesalius.reader.dicom_reader as dcm
from invesalius import utils

CORONAL = 2

SURFACE_QUALITY = {
    _("Low"): (3, 2, 0.3000, 0.4),
    _("Medium"): (2, 2, 0.3000, 0.4),
    _("High"): (0, 1, 0.3000, 0.1),
    _("Optimal *"): (0, 2, 0.3000, 0.4)}


def UpdateThresholdModes(scalar_range):
    thresh_min, thresh_max = scalar_range
    presets_list = (self.thresh_ct, self.thresh_mri)

    for presets in presets_list:
        for key in presets:
            t_min, t_max = presets[key]

            if t_min or t_max:  # setting custom preset
                t_min = thresh_min
                t_max = thresh_max
            if t_min < thresh_min:
                t_min = thresh_min
            if t_max > thresh_max:
                t_max = thresh_max

            # This has happened in Analyze files
            # TODO: find a good solution for presets in Analyze files
            if t_min > thresh_max:
                t_min = thresh_min
            if t_max < thresh_min:
                t_max = thresh_max

            presets[key] = (t_min, t_max)

        # TODO: sendMessage


def OpenDicomGroup(dicom_group, interval, file_range):
    # Retrieve general DICOM headers
    dicom_ = dicom_group.GetDicomSample()

    # Create image data
    interval += 1
    file_list = dicom_group.GetFilenameList()[::interval]
    if not file_list:
        file_list = [i.image.file for i in dicom_group.GetHandSortedList()[::interval]]

    if file_range and file_range[1] > file_range[0]:
        file_list = file_list[file_range[0]:file_range[1] + 1]

    z_spacing = dicom_group.zspacing * interval

    size = dicom_.image.size
    bits = dicom_.image.bits_allocad
    x_y_spacing = dicom_.image.spacing
    orientation = dicom_.image.orientation_label

    sx, sy = size
    n_slices = len(file_list)
    resolution_percentage = utils.calculate_resizing_tofitmemory(int(sx), int(sy), n_slices, bits / 8)

    wl = float(dicom_.image.level)
    ww = float(dicom_.image.window)
    matrix_, scalar_range, filename_ = image_utils.dcm2memmap(file_list, size, orientation, resolution_percentage)

    slice_ = sl.Slice()
    slice_.matrix = matrix_
    slice_.matrix_filename = filename_

    if orientation == 'AXIAL':
        slice_.spacing = x_y_spacing[0], x_y_spacing[1], z_spacing
    elif orientation == 'CORONAL':
        slice_.spacing = x_y_spacing[0], z_spacing, x_y_spacing[1]
    elif orientation == 'SAGITTAL':
        slice_.spacing = z_spacing, x_y_spacing[1], x_y_spacing[0]

    # 1(a): Fix gantry tilt, if any
    tilt_value = dicom_.acquisition.tilt
    if tilt_value:
        tilt_value = -1 * tilt_value
        image_utils.FixGantryTilt(matrix_, slice_.spacing, tilt_value)

    slice_.window_level = wl
    slice_.window_width = ww

    scalar_range = int(matrix_.min()), int(matrix_.max())

    UpdateThresholdModes(scalar_range)

    return matrix_, filename_, dicom_


def create_mask(matrix, threshold_range):
    mask_ = mask.Mask()
    mask_.create_mask(matrix.shape)
    mask_.threshold_range = threshold_range
    return mask_


def get_poly_data(slice_, mask_, surface_parameters):
    matrix = slice_.matrix
    filename_img = slice_.matrix_filename
    spacing = slice_.spacing

    algorithm = surface_parameters['method']['algorithm']
    options = surface_parameters['method']['options']

    quality = surface_parameters['options']['quality']
    fill_holes = surface_parameters['options']['fill']
    keep_largest = surface_parameters['options']['keep_largest']

    mode = 'CONTOUR'  # 'GRAYSCALE'
    min_value, max_value = mask_.threshold_range

    mask_.matrix.flush()

    if quality in SURFACE_QUALITY.keys():
        image_data_resolution = SURFACE_QUALITY[quality][0]
        smooth_iterations = SURFACE_QUALITY[quality][1]
        smooth_relaxation_factor = SURFACE_QUALITY[quality][2]
        decimate_reduction = SURFACE_QUALITY[quality][3]
    else:
        image_data_resolution = None
        smooth_iterations = None
        smooth_relaxation_factor = None
        decimate_reduction = None

    pipeline_size = 4
    if decimate_reduction:
        pipeline_size += 1
    if smooth_iterations and smooth_relaxation_factor:
        pipeline_size += 1
    if fill_holes:
        pipeline_size += 1
    if keep_largest:
        pipeline_size += 1

    flip_image = True

    n_processors = multiprocessing.cpu_count()

    pipe_in, pipe_out = multiprocessing.Pipe()
    o_piece = 1
    piece_size = 2000

    n_pieces = int(round(matrix.shape[0] / piece_size + 0.5, 0))

    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()

    p = []
    for _ in xrange(n_processors):
        sp = surface_process.SurfaceProcess(pipe_in, filename_img,
                                            matrix.shape, matrix.dtype,
                                            mask_.temp_file,
                                            mask_.matrix.shape,
                                            mask_.matrix.dtype,
                                            spacing,
                                            mode, min_value, max_value,
                                            decimate_reduction,
                                            smooth_relaxation_factor,
                                            smooth_iterations, None,
                                            flip_image, q_in, q_out,
                                            algorithm != 'Default',
                                            algorithm,
                                            image_data_resolution)
        p.append(sp)
        sp.start()

    for i in xrange(n_pieces):
        init = i * piece_size
        end = init + piece_size + o_piece
        roi = slice(init, end)
        q_in.put(roi)
        print "new_piece", roi

    for _ in p:
        q_in.put(None)

    none_count = 1
    while 1:
        msg = pipe_out.recv()
        if msg is None:
            none_count += 1

        if none_count > n_pieces:
            break

    # noinspection PyUnresolvedReferences
    poly_data_append = vtk.vtkAppendPolyData()
    t = n_pieces
    while t:
        filename_poly_data = q_out.get()

        # noinspection PyUnresolvedReferences
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename_poly_data)
        reader.Update()

        poly_data = reader.GetOutput()

        poly_data_append.AddInputData(poly_data)
        del reader
        del poly_data
        t -= 1

    poly_data_append.Update()
    poly_data = poly_data_append.GetOutput()
    del poly_data_append

    if algorithm == 'ca_smoothing':
        # noinspection PyUnresolvedReferences
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly_data)
        normals.ComputeCellNormalsOn()
        normals.Update()
        del poly_data
        poly_data = normals.GetOutput()
        del normals

        # noinspection PyUnresolvedReferences
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly_data)
        clean.PointMergingOn()
        clean.Update()

        del poly_data
        poly_data = clean.GetOutput()
        del clean

        mesh = cy_mesh.Mesh(poly_data)
        # noinspection PyTypeChecker
        cy_mesh.ca_smoothing(mesh, options['angle'], options['max distance'], options['min weight'], options['steps'])

        # noinspection PyUnresolvedReferences
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(poly_data)
        writer.SetFileName('/tmp/ca_smoothing_inv.ply')
        writer.Write()
    else:
        # noinspection PyUnresolvedReferences
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(poly_data)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetRelaxationFactor(smooth_relaxation_factor)
        smoother.SetFeatureAngle(80)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOn()
        smoother.Update()
        del poly_data
        poly_data = smoother.GetOutput()
        del smoother

    if decimate_reduction:
        # noinspection PyUnresolvedReferences
        decimation = vtk.vtkQuadricDecimation()
        decimation.SetInputData(poly_data)
        decimation.SetTargetReduction(decimate_reduction)
        decimation.Update()
        del poly_data
        poly_data = decimation.GetOutput()
        del decimation

    if keep_largest:
        # noinspection PyUnresolvedReferences
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(poly_data)
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        del poly_data
        poly_data = conn.GetOutput()
        del conn

    # Filter used to detect and fill holes. Only fill boundary edges holes.
    if fill_holes:
        # noinspection PyUnresolvedReferences
        filled_poly_data = vtk.vtkFillHolesFilter()
        filled_poly_data.SetInputData(poly_data)
        filled_poly_data.SetHoleSize(300)
        filled_poly_data.Update()
        del poly_data
        poly_data = filled_poly_data.GetOutput()
        del filled_poly_data

    # noinspection PyUnresolvedReferences
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly_data)
    normals.SetFeatureAngle(80)
    normals.AutoOrientNormalsOn()
    normals.Update()
    del poly_data
    poly_data = normals.GetOutput()

    # Improve performance
    # noinspection PyUnresolvedReferences
    stripper = vtk.vtkStripper()
    stripper.SetInputData(poly_data)
    stripper.PassThroughCellIdsOn()
    stripper.PassThroughPointIdsOn()
    stripper.Update()
    del poly_data

    poly_data = stripper.GetOutput()

    return poly_data


def main(dicom_directory, path_prefix):
    patients_groups = dcm.GetDicomGroups(dicom_directory)
    group = dcm.SelectLargerDicomGroup(patients_groups)
    matrix, matrix_filename, dicom = OpenDicomGroup(group, 0, [0, 0])

    # ##########################

    for __ in [1, 2, 3]:  # TODO: for each preset
        threshold_range = None  # TODO

        mask_ = create_mask(_, threshold_range)  # TODO

        # ##########################

        surface_options = {
            'method': {
                'algorithm': 'Default',
                'options': {},
            },
            'options': {
                'quality': _('Optimal *'),
                'fill': False,
                'keep_largest': False,
                'overwrite': False,
            },
        }
        poly_data = get_poly_data(_, mask_, surface_options)  # TODO

        # noinspection PyUnresolvedReferences
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
        writer.SetFileName('{}-{}.stl'.format(path_prefix, None))  # TODO
        writer.SetInputData(poly_data)
        writer.Write()

if __name__ == '__main__':
    main(_, _)  # TODO: ask arguments
