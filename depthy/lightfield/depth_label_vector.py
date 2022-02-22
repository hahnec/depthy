import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def create_angle_masks(labels):

    angle_labels = np.arctan(labels) * 180 / np.pi
    angle_probes = np.array([-1, 0, 1, 2])*45
    masks = np.zeros([2, len(angle_labels), 3, 3])

    for i in range(len(angle_labels)):

        # determine pixels closest to label
        angle_distances = angle_probes-angle_labels[i]
        angle_near_idx = np.argmin(abs(angle_distances))
        angle_distance = angle_probes[angle_near_idx]-angle_labels[i]
        angle_adja_idx = angle_near_idx-1 if angle_distance > 0 else angle_near_idx+1

        # assign weights to labels closest to pixels
        grad_weights = np.zeros(len(angle_probes))
        grad_weights[angle_adja_idx % 4] = abs(angle_distance)/45
        grad_weights[angle_near_idx % 4] = 1-grad_weights[angle_adja_idx % 4]

        # create mask for current label
        masks[0, i, 0, :3] = grad_weights[:3]
        if angle_adja_idx == -1:
            masks[0, i, 1, 0] = grad_weights[-1]
        else:
            masks[0, i, 1, -1] = grad_weights[-1]

        # rotate mask by 180 degrees for opposite direction
        masks[1, i, ...] = masks[0, i, ...][::-1, ::-1]

    return masks


def set_binary_maps(cost_maps):

    binary_maps = np.zeros_like(cost_maps, dtype=bool)
    u, v = np.ogrid[:cost_maps.shape[1], :cost_maps.shape[2]]
    binary_maps[np.argmin(cost_maps, axis=0), u, v] = 1

    return binary_maps


def local_constraint_regularizer(labels, binary_maps, masks, reg_v=None):

    # init regularization term
    reg_v = np.zeros(binary_maps.shape, dtype=np.float64) if reg_v is None else reg_v

    # compute winning labels
    label_map = np.argmax(binary_maps, axis=0)

    # compute gradients in both slope directions
    slide_map = sliding_window_view(labels[label_map], window_shape=(3, 3), axis=(0, 1))
    slide_map = np.pad(slide_map, pad_width=((1, 1), (1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    maskp = masks[0, label_map]
    maskn = masks[1, label_map]
    dir_p = np.einsum('ijkl,ijkl->ij', slide_map, maskp)
    dir_n = np.einsum('ijkl,ijkl->ij', slide_map, maskn)
    gradp = labels[label_map] - dir_p
    gradn = labels[label_map] - dir_n

    # set punishment parameters
    label_gap = np.min(np.abs(np.diff(labels)))
    pen = binary_maps * label_gap * 10e5
    tol = label_gap / 1

    # harsh punishment for change to lower disparities (occlusion constraint)
    reg_p = 50*pen * np.array(gradp < 0)
    reg_n = 50*pen * np.array(gradn < 0)

    # punishment if local neighbour in direction of slope varies (slope consistency constraint)
    reg_p = reg_p + np.array(np.abs(gradp) > tol)*1e-1*0
    reg_n = reg_n + np.array(np.abs(gradn) > tol)*1e-1*0

    # bi-directional slope deviation constraint
    idx_v = (reg_p != 0) & (reg_n != 0)

    # merge directional gradients
    reg_v[idx_v] += (reg_p[idx_v] + reg_n[idx_v])/2

    return reg_v


def get_labels(local_disp=None, label_num: int = 9, label_method: str = 'hist'):

    min_disp = np.min(local_disp)
    max_disp = np.max(local_disp)

    # use pre-determined label method based on angles (if no disparities are provided as a reference)
    label_method = 'angl' if local_disp is None else label_method

    # define quantized label values
    if label_method is None:
        labels = sorted(local_disp.unique())
    elif label_method == 'disp':
        dispar = np.linspace(min_disp, max_disp, label_num)
        angles = np.arctan(1/dispar) / np.pi * 180
        labels = np.tan(angles / 180 * np.pi)
    elif label_method == 'angl':
        leq_45 = 45*(np.linspace(0, 1, 3)**.5-1)
        geq_45 = 45*np.linspace(0, 1, label_num-2)**2*2
        geq_45 /= geq_45[np.argmin(np.abs(geq_45-45))]/45  # normalize so that angle 45° @ d=1 is among set
        geq_45[geq_45 > 90] = 90    # clip values larger than 90 degrees
        angles = np.concatenate([leq_45, geq_45[1:]])
        labels = np.tan(angles / 180 * np.pi)
    else:
        pdf, labels = np.histogram(local_disp, range=(min_disp, max_disp), bins=label_num)

    return labels


def local_label_optimization(local_disp, coherence, max_iter=100, perc=1, labels=None):

    # exclude outlying labels
    min_disp = np.percentile(local_disp, perc)
    max_disp = np.percentile(local_disp, 100 - perc)
    local_disp[local_disp > max_disp] = max_disp
    local_disp[local_disp < min_disp] = min_disp

    # reduce channel dimension for performance
    local_disp = np.mean(local_disp, axis=-1) if len(local_disp.shape) == 3 else local_disp
    coherence = np.mean(coherence, axis=-1) if len(coherence.shape) == 3 else coherence

    # determine depth labels if not provided
    labels = get_labels(local_disp) if labels is None else labels

    # use masks for sliding window local gradient slope analysis
    masks = create_angle_masks(labels)

    # remove zeros in coherence
    coherence += np.min(coherence)

    # local cost volume
    cost_volume = coherence * np.abs(labels[:, None, None] * np.ones((len(labels),)+local_disp.shape) - local_disp)

    # initialize label map (binary indicator functions)
    init_maps = set_binary_maps(cost_volume)
    binary_maps = init_maps.copy()

    energy_list = []
    regularizer = np.zeros(cost_volume.shape, dtype=np.float64)
    for i in range(max_iter):

        # get local constraint
        regularizer = local_constraint_regularizer(labels, binary_maps, masks, regularizer)

        # update label maps
        binary_maps = set_binary_maps(cost_volume+regularizer)

        idx_map = np.abs(np.argmax(binary_maps, axis=0) - np.argmax(init_maps, axis=0)) > 0
        regularizer[:, ~idx_map] = 0

        # compute energy (error) which we aim to minimize
        energy = np.sum((cost_volume+regularizer)*binary_maps)
        energy_list.append(energy)

    # reduce dimension across binary indicator functions to map of labels
    cons_map = labels[np.argmax(binary_maps, axis=0)]

    # expand channel dimension if reduced earlier
    cons_map = np.repeat(cons_map[..., None], 3, axis=-1) if len(cons_map.shape) == 2 else cons_map

    return cons_map
