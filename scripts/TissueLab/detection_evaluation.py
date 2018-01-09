import time

import numpy as np
from hungarian import lap
from openalea.container import array_dict
from scipy.cluster.vq import vq


def evaluate_nuclei_detection(nuclei_topomesh, ground_truth_topomesh,
                              max_matching_distance=3.0, outlying_distance=5.0,
                              max_distance=100.):
    """

    Parameters
    ----------
    nuclei_topomesh : topomesh
        a "vertex topomesh" with nuclei/seed coordinates (require wisp_property
        'barycenter') to compare to a ground truth
    ground_truth_topomesh : topomesh
        a "vertex topomesh" with expertised nuclei/seed coordinates (require
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
    nuc_ids = nuclei_topomesh.wisp_property('barycenter', 0).keys()
    nuc_coords = nuclei_topomesh.wisp_property('barycenter', 0).values()
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
    evaluation['False Positive'] = nuclei_topomesh.nb_wisps(0) - segmentation_outliers.values().sum() - evaluation['True Positive']

    evaluation['Precision'] = evaluation['True Positive'] / float(evaluation['True Positive'] + evaluation['False Positive']) if evaluation['True Positive'] + evaluation['False Positive'] > 0 else 100.
    evaluation['Recall'] = evaluation['True Positive'] / float(evaluation['True Positive'] + evaluation['False Negative'])
    evaluation['Jaccard'] = evaluation['True Positive'] / float(evaluation['True Positive'] + evaluation['False Positive'] + evaluation['False Negative'])
    evaluation['Dice'] = 2. * evaluation['True Positive'] / float(2. * evaluation['True Positive'] + evaluation['False Positive'] + evaluation['False Negative'])

    print "Precision ", np.round(100. * evaluation['Precision'], 2), "%, Recall ", np.round(100. * evaluation['Recall'], 2), "%"

    return evaluation
