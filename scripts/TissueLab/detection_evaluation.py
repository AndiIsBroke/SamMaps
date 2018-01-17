import numpy as np
import scipy.ndimage as nd

try:
    from hungarian import lap
except ImportError:
    inst_mess = ""
    inst_mess += '\ngit clone https://github.com/hrldcpr/hungarian'
    inst_mess += '\ncd hungarian'
    inst_mess += "\n# If under a conda environment, activate it and use '--prefix=$CONDA_PREFIX' instead of '--user'"
    inst_mess += '\npython setup.py install --user'
    raise ImportError("Please install the 'hungarian' library: {}".format(inst_mess))

from time import time
from copy import deepcopy
from scipy.cluster.vq import vq

from openalea.container import array_dict
from timagetk.wrapping.bal_trsf import BalTransformation


def evaluate_positions_detection(vertex_topomesh, ground_truth_topomesh,
                              max_matching_distance=3.0, outlying_distance=5.0,
                              max_distance=100.):
    """

    Parameters
    ----------
    vertex_topomesh : topomesh
        a topomesh with nuclei/seed coordinates (require wisp_property
        'barycenter') to compare to a ground truth
    ground_truth_topomesh : topomesh
        a topomesh with expertised nuclei/seed coordinates (require
        wisp_property 'barycenter') to be used as reference for evaluation
    max_matching_distance : float

    outlying_distance : float, optional

    max_distance : float, optional


    Returns
    -------
    dictionary of indicators (after matching) such as:
        'True Positive':
        'False Negative':
        'False Positive':
        'Precision':
        'Recall':
        'Jaccard':
        'Dice':

    Notes
    -----
    Requires the Hungarian library : https://github.com/hrldcpr/hungarian
    """
    nuc_ids = vertex_topomesh.wisp_property('barycenter', 0).keys()
    nuc_coords = vertex_topomesh.wisp_property('barycenter', 0).values()
    grt_coords = ground_truth_topomesh.wisp_property('barycenter', 0).values()

    segmentation_ground_truth_matching = vq(nuc_coords, grt_coords)
    ground_truth_segmentation_complete_matching = np.array([vq(nuc_coords, np.array([p]))[1] for p in grt_coords])

    segmentation_outliers = array_dict(segmentation_ground_truth_matching[1] > outlying_distance + 1, nuc_ids)

    cost_matrix = deepcopy(ground_truth_segmentation_complete_matching)
    if cost_matrix.shape[0] < cost_matrix.shape[1]:
        cost_matrix = np.concatenate([cost_matrix, np.ones((cost_matrix.shape[1] - cost_matrix.shape[0], cost_matrix.shape[1])) * max_distance])
    elif cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = np.concatenate([cost_matrix, np.ones((cost_matrix.shape[0], cost_matrix.shape[0] - cost_matrix.shape[1])) * max_distance], axis=1)

    cost_matrix[cost_matrix > outlying_distance] = max_distance

    initial_cost_matrix = deepcopy(cost_matrix)

    start_time = time()
    print "--> Hungarian assignment..."
    assignment = lap(cost_matrix)
    end_time = time()
    print "<-- Hungarian assignment     [", end_time - start_time, "s]"

    ground_truth_assignment = np.arange(ground_truth_topomesh.nb_wisps(0))
    segmentation_assignment = assignment[0][:ground_truth_topomesh.nb_wisps(0)]
    assignment_distances = initial_cost_matrix[
        (ground_truth_assignment, segmentation_assignment)]
    # print "Assignment : ",assignment_distances.mean()

    evaluation = {}
    evaluation['True Positive'] = (assignment_distances < max_matching_distance).sum()
    evaluation['False Negative'] = (assignment_distances >= max_matching_distance).sum()
    evaluation['False Positive'] = vertex_topomesh.nb_wisps(0) - segmentation_outliers.values().sum() - evaluation['True Positive']

    evaluation['Precision'] = evaluation['True Positive'] / float(evaluation['True Positive'] + evaluation['False Positive']) if evaluation['True Positive'] + evaluation['False Positive'] > 0 else 100.
    evaluation['Recall'] = evaluation['True Positive'] / float(evaluation['True Positive'] + evaluation['False Negative'])
    evaluation['Jaccard'] = evaluation['True Positive'] / float(evaluation['True Positive'] + evaluation['False Positive'] + evaluation['False Negative'])
    evaluation['Dice'] = 2. * evaluation['True Positive'] / float(2. * evaluation['True Positive'] + evaluation['False Positive'] + evaluation['False Negative'])

    print "Precision ", np.round(100. * evaluation['Precision'], 2), "%, Recall ", np.round(100. * evaluation['Recall'], 2), "%"

    return evaluation

def get_biggest_bounding_box(bboxes):
    """
    Compute the bounding box "size" and return the label for the largest.

    Parameters
    ----------
    bboxes : dict
        dictionary of bounding box (values) with labels as keys

    Returns
    -------
    label : int
        the labelwith the largest bounding box
    """
    label_biggest_bbox = None
    bbox_size = 0
    is2D = len(bboxes.values()[0])==2
    for label, bbox in bboxes.items():
        if is2D:
            x_sl, y_sl = bbox
            size = (x_sl.stop - x_sl.start) * (y_sl.stop - y_sl.start)
        else:
            x_sl, y_sl, z_sl = bbox
            size = (x_sl.stop - x_sl.start) * (y_sl.stop - y_sl.start) * (z_sl.stop - z_sl.start)
        if bbox_size < size:
            bbox_size = size
            label_biggest_bbox = label

    return label_biggest_bbox


def get_background_value(seg_im, microscope_orientation=1):
    """
    Determine the background value using the largewt bounding box.

    Parameters
    ----------
    seg_im : SpatialImage
        SpatialImage for which to determine the background value
    microscope_orientation : int
        For upright microscope use '1' for inverted use (-1)

    Returns
    -------
    background : int
        the labelwith the largest bounding box
    """
    if microscope_orientation == -1:
        top_slice = seg_im[:,:,0]
    else:
        top_slice = seg_im[:,:,-1]
    top_slice_labels = sorted(np.unique(top_slice))
    top_bboxes = nd.find_objects(top_slice, max_label = top_slice_labels[-1])
    top_bboxes = {n+1: top_bbox for n, top_bbox in enumerate(top_bboxes) if top_bbox is not None}

    return get_biggest_bounding_box(top_bboxes)


def apply_trsf2pts(rigid_trsf, points):
    """
    Function applying a RIGID transformation to a set of points.

    Parameters
    ----------
    rigid_trsf: np.array | BalTransformation
        a quaternion obtained by rigid registration
    points: np.array
        a Nxd list of points to tranform, with d the dimensionality and N the
        number of points
    """
    if isinstance(rigid_trsf, BalTransformation):
        try:
            assert rigid_trsf.isLinear()
        except:
            raise TypeError("The provided transformation is not linear!")
        rigid_trsf = rigid_trsf.mat.to_np_array()
    X, Y, Z = points.T
    homogeneous_points = np.concatenate([np.transpose([X,Y,Z]), np.ones((len(X),1))], axis=1)
    transformed_points = np.einsum("...ij,...j->...i", rigid_trsf, homogeneous_points)

    return transformed_points[:,:3]


def filter_topomesh_vertices(topomesh, vtx_list="L1"):
    """
    Return a filtered topomesh containing only the values found in `vtx_list`.

    Parameters
    ----------
    topomesh : vertex_topomesh
        a topomesh to edit
    vtx_list : str | list
        if a list, the ids it contains will be used to filter the `topomesh`
        can be a string like "L1", then propery "layer" should exists!

    Returns
    -------
    vertex_topomesh
    """
    if isinstance(vtx_list, str):
        try:
            assert "layer" in list(topomesh.wisp_property_names(0))
        except AssertionError:
            raise ValueError("Property 'layer' is missing in the topomesh!")
    # - Duplicate the topomesh:
    filtered_topomesh = deepcopy(topomesh)
    # - Define selected vertices:
    if vtx_list == "L1":
        # -- Filter L1 seeds from 'detected_topomesh':
        filtered_cells = np.array(list(filtered_topomesh.wisps(0)))[filtered_topomesh.wisp_property('layer',0).values()==1]
    elif vtx_list == "L2":
        # -- Filter L2 seeds from 'detected_topomesh':
        filtered_cells = np.array(list(filtered_topomesh.wisps(0)))[filtered_topomesh.wisp_property('layer',0).values()==2]
    elif isinstance(vtx_list, list):
        filtered_cells = [v for v in vtx_list if v in filtered_topomesh.wisps(0)]
    else:
        raise ValueError("Unable to use given `vtx_list`, please check it!")
    # - Remove unwanted vertices:
    vtx2remove = list(set(filtered_topomesh.wisps(0)) - set(filtered_cells))
    for c in vtx2remove:
        filtered_topomesh.remove_wisp(0,c)
    # - Update properies found in the original topomesh:
    for ppty in filtered_topomesh.wisp_property_names(0):
        vtx = list(filtered_topomesh.wisps(0))
        ppty_dict = array_dict(filtered_topomesh.wisp_property(ppty, 0).values(vtx), keys=vtx)
        filtered_topomesh.update_wisp_property(ppty, 0, ppty_dict)

    return filtered_topomesh
