import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage import transform
from skimage.registration._optical_flow_utils import get_warp_points
from itertools import permutations
from functools import partial

from ._ilk import partial_flow_ilk
from ..filters import masked_filter


# TODO: ensure that integrated intensity remains constant
def warp(x, flow, intensity_flow=False, **kw):
    grid = np.meshgrid(*[np.arange(n, dtype=x.dtype) for n in x.shape],
                       indexing='ij', sparse=True)
    if intensity_flow:
        print(flow.shape)
        x_warp = transform.warp(x, get_warp_points(grid, flow[:-1]), mode='edge', **kw)
        return x_warp + flow[-1]
    else:
        return transform.warp(x, get_warp_points(grid, flow), mode='edge', **kw)


def warp_flow(flow_in, flow_warp):
    """
    Warp a flow field flow_in by another flow field (flow_warp) such that the flow field moves
    in concert with the image that would be warped by flow_warp.
    Note that flow is an inverse coordinate map (transforms output coordinates to input coordinates),
    so the flow field has to be warped in the opposite direction of the image
    :param flow_in: flow field to warp
    :param flow_warp: flow field that warps flow_in
    :return:
    """
    flow_out = np.empty_like(flow_in)
    for i, f in enumerate(flow_in):
        flow_out[i] = warp(f, -flow_warp)

    return flow_out


def sum_flows(flow_sequence, reverse=False):
    if reverse:
        # Reverse the sequence
        flow_sequence = [reverse_flow(f) for f in flow_sequence[::-1]]

    # Starting from destination frame, work backwards to source frame
    rev_seq = flow_sequence[::-1]
    tot_flow = rev_seq[0].copy()
    for i, f in enumerate(rev_seq[1:]):
        # At each step, need to warp the current flow by the total flow from current frame to destination frame
        tot_flow += warp_flow(f, -tot_flow)

    return tot_flow


def reverse_flow(flow):
    """
    Reverse a flow field to give the (approximate) inverse transform.
    Note that flow is an inverse coordinate map (transforms output coordinates to input coordinates),
    so the flow field has to be warped in the opposite direction of the image
    :param flow:
    :return:
    """
    rev = np.empty_like(flow)
    for i, f in enumerate(flow):
        rev[i] = -warp(f, -flow)

    return rev


def blend_frames(reference_image, moving_image, sigma, num=2, replace=False):
    output = np.empty((num + 2, *reference_image.shape))
    output[0] = reference_image
    output[-1] = moving_image

    weights = np.zeros_like(output)
    weights[0] = 1
    weights[-1] = 1
    blended = masked_filter(output, weights, gaussian_filter1d, axis=0, sigma=sigma, mode='nearest')

    if replace:
        return blended
    else:
        # Don't filter the original images
        output[1:-1] = blended[1:-1]
        return output


def partial_flow_blended(reference_image, moving_image, flow_axes,
                         blend_sigma=1, replace=False, blend_num=2, momentum=True, momentum_radius=1,
                         post_blend_op=None, op_kwargs=None,
                         radius=7, sigma=None, num_warp=10, gaussian=False,
                         prefilter=False, weights=None, update_weights=False, intensity_flow=False,
                         dtype=np.float32):
    # Make blended frames
    blended = blend_frames(reference_image, moving_image, blend_sigma, blend_num, replace=replace)

    if post_blend_op is not None:
        if op_kwargs is None:
            op_kwargs = {}
        blended = post_blend_op(blended, **op_kwargs)

    if momentum:
        # Shift flow axes to account for blend axis
        flow_axes = tuple([ax + 1 if ax >= 0 else ax for ax in flow_axes])

        # Add momentum radius
        if np.isscalar(radius):
            radius = [radius] * np.ndim(moving_image)
        radius = [momentum_radius] + list(radius)

        # print(radius, flow_axes)
        # print(np.any(np.isnan(blended)))

        blend_flow = partial_flow_ilk(blended[:-1], blended[1:], flow_axes=flow_axes, radius=radius, sigma=sigma,
                                      num_warp=num_warp, gaussian=gaussian, prefilter=prefilter, weights=weights,
                                      update_weights=update_weights, intensity_flow=intensity_flow, dtype=dtype)
        # print(blend_flow.shape)
        # print([np.any(np.isnan(bf)) for bf in blend_flow])
        flow = np.array([np.sum(f, axis=0) for f in blend_flow[1:]])

        return flow
    else:
        blend_flow = []
        for i in range(len(blended) - 1):
            bf = partial_flow_ilk(blended[i], blended[i + 1], flow_axes=flow_axes, radius=radius, sigma=sigma,
                                  num_warp=num_warp, gaussian=gaussian, prefilter=prefilter, weights=weights,
                                  update_weights=update_weights, intensity_flow=intensity_flow, dtype=dtype)
            blend_flow.append(bf)
        # print(np.array(blend_flow).shape)
        flow = np.sum(blend_flow, axis=0)
        return flow


def bidirectional_flow(reference_image, moving_image, *, flow_axes,
                       radius=7, sigma=None, num_warp=10, gaussian=False,
                       prefilter=False, weights=None, update_weights=False, intensity_flow=False,
                       dtype=np.float32):
    """
    Solve optical flow in both directions and return the mean
    Should return similar results to partial_flow_ilk, but with less variability
    :param reference_image:
    :param moving_image:
    :param flow_axes:
    :param radius:
    :param sigma:
    :param num_warp:
    :param gaussian:
    :param prefilter:
    :param weights:
    :param update_weights:
    :param intensity_flow:
    :param dtype:
    :return:
    """
    fwd = partial_flow_ilk(reference_image, moving_image, flow_axes=flow_axes, radius=radius, sigma=sigma,
                           num_warp=num_warp, gaussian=gaussian, prefilter=prefilter, weights=weights,
                           update_weights=update_weights, intensity_flow=intensity_flow, dtype=dtype)

    rev = partial_flow_ilk(moving_image, reference_image, flow_axes=flow_axes, radius=radius, sigma=sigma,
                           num_warp=num_warp, gaussian=gaussian, prefilter=prefilter, weights=weights,
                           update_weights=update_weights, intensity_flow=intensity_flow, dtype=dtype)

    rev_trans = reverse_flow(rev)

    return 0.5 * (fwd + rev_trans)


# def align_to_ref(image, flows, ref_index, axis=0, return_flows=False, upscale=None, **warp_kw):
#     """
#     Align (N-1)D slices of image along axis to a reference slice
#     :param ndarray image: ND image
#     :param ndarray flows: ND flows along each dimension
#     :param int ref_index: index of slice to which to align
#     :param int axis: axis along which to align slices
#     :param bool return_flows: if True, return aligning flows for each slice
#     :return:
#     """
#     if upscale is not None:
#         if np.isscalar(upscale):
#             upscale = tuple([upscale if i != axis else 1 for i in range(np.ndim(image))])
#         shape = tuple([si * ui for si, ui in zip(image.shape, upscale)])
#         aligned_image = np.empty(shape, dtype=image.dtype)
#     else:
#         aligned_image = np.empty_like(image)
#
#     swap = aligned_image.swapaxes(0, axis)
#     # Exclude flow along axis
#     axis_flows = [f for i, f in enumerate(flows) if i != axis]
#     align_flows = []
#     for i in range(image.shape[axis]):
#         orig = np.take(image, i, axis=axis)
#
#         # print(output_shape)
#
#         if i <= ref_index:
#             # Warp low to high (forward)
#             if i == ref_index:
#                 warped = orig.copy()
#                 flow_sum = None
#             else:
#                 flow_stack = np.take(axis_flows, np.arange(i, ref_index), axis=axis + 1)
#                 # print(flow_stack.shape)
#                 # flow_sum = np.sum(flow_stack, axis=axis + 1)
#                 flow_stack = np.swapaxes(flow_stack, 0, axis + 1)
#                 flow_sum = sum_flows(flow_stack)
#                 # print(flow_sum.shape)
#
#                 warped = warp(orig, flow_sum, **warp_kw)
#         else:
#             # Warp high to low (reverse)
#             flow_stack = np.take(axis_flows, np.arange(ref_index, i), axis=axis + 1)
#             # flow_sum = -np.sum(flow_stack, axis=axis + 1)
#
#             flow_stack = np.swapaxes(flow_stack, 0, axis + 1)
#             flow_sum = sum_flows(flow_stack, reverse=True)
#
#             warped = warp(orig, flow_sum, **warp_kw)
#
#         if upscale is not None:
#             output_shape = swap[i].shape
#             # warped = transform.pyramid_expand(warped, upscale, mode='reflect')
#             warped = transform.resize(warped, output_shape)
#             if flow_sum is not None:
#                 # flow_sum = transform.pyramid_expand(flow_sum, upscale, mode='reflect')
#                 flow_sum = transform.resize(flow_sum, (flow_sum.shape[0],) + output_shape)
#                 # Multiply flow in each dimension by corresponding upscale factor
#                 upscale_stack = [uj for j, uj in enumerate(upscale) if j != axis]
#                 for j, uj in enumerate(upscale_stack):
#                     flow_sum[j] *= uj
#
#         swap[i] = warped
#         align_flows.append(flow_sum)
#
#     if return_flows:
#         return aligned_image, align_flows
#     else:
#         return aligned_image
#
#
# def unwarp_from_ref(aligned_image, align_flows, axis=0, downscale=None, **warp_kw):
#     if downscale is not None:
#         if np.isscalar(downscale):
#             shape = tuple([int(si / downscale) if i != axis else si for i, si in enumerate(aligned_image.shape)])
#         else:
#             shape = tuple([int(si / di) for si, di in zip(aligned_image.shape, downscale)])
#         image = np.empty(shape, dtype=aligned_image.dtype)
#     else:
#         image = np.empty_like(aligned_image)
#
#     # image = np.empty_like(aligned_image)
#     swap = image.swapaxes(0, axis)
#     for i in range(aligned_image.shape[axis]):
#         warped = np.take(aligned_image, i, axis=axis)
#
#         if align_flows[i] is not None:
#             print(warped.shape, align_flows[i].shape)
#             unwarped = warp(warped, -align_flows[i] / 2, **warp_kw)
#         else:
#             unwarped = warped.copy()
#
#         if downscale is not None:
#             output_shape = swap[i].shape
#             # unwarped = transform.pyramid_reduce(unwarped, downscale, mode='reflect')
#             unwarped = transform.resize(unwarped, output_shape, anti_aliasing=False)
#
#         swap[i] = unwarped
#
#     return image


# def warp_step_cost(group_exists, end, direction, axis):
#
#     end_exists = np.take(group_exists, end, axis)
#     if group_exists.shape[axis] > end + direction > -1:
#         next_exists = np.take(group_exists, end + 1, axis)
#     else:
#         next_exists = np.zeros_like(end_exists)
#
#     # Standard cost is 1
#     cost = np.ones(end_exists.shape)
#     # If there is no group to warp to, cost is 2
#     cost[~end_exists] = 2
#     # If there is no group to warp to and no group one step farther, cannot warp
#     cost[~end_exists & ~next_exists] = np.inf
#
#     return cost

def warp_step_cost(group_exists, start, direction, axis):
    if group_exists.shape[axis] > start[axis] + direction > -1:
        end = list(start).copy()
        end[axis] += direction
        end_exists = group_exists[tuple(end)]
        print(end_exists)
        next_coords = list(end).copy()
        next_coords[axis] += direction
        if group_exists.shape[axis] > next_coords[axis] > -1:
            next_exists = group_exists[tuple(next_coords)]
        else:
            next_exists = False

        if end_exists:
            return 1
        elif next_exists:
            return 2
        else:
            return np.inf
    else:
        return np.inf


def warp_path_cost(group_exists, start_coords, end_coords, axis_order):
    cost = 0

    # Convert to list to allow updates
    start_coords = list(start_coords)
    start_coord_list = []

    for axis in axis_order:
        # print(axis, start_coords)
        start = start_coords[axis]
        end = end_coords[axis]
        if end != start:
            direction = np.sign(end - start)
            # costs = np.array([warp_step_cost(group_exists, e, direction, axis) for e in np.arange(start + 1, end + 1)])
            # sub_start = tuple(c for i, c in enumerate(start_coords) if i != axis)

            if axis < group_exists.ndim - 1:
                step_starts = [start_coords[:axis] + [s] + start_coords[axis + 1:] for s in np.arange(start, end)]
            else:
                step_starts = [start_coords[:axis] + [s] for s in np.arange(start, end)]

            print(step_starts)
            costs = [warp_step_cost(group_exists, tuple(ss), direction, axis) for ss in step_starts]

            start_coord_list += step_starts

            cost += np.sum(costs)

            # Update start position
            start_coords[axis] = end_coords[axis]

    return cost, start_coord_list


def solve_warp_axis_order(group_exists, start_coords, end_coords):
    axes = np.arange(group_exists.ndim)
    axis_orders = list(permutations(axes))
    costs = np.empty(len(axis_orders))
    coord_history = []

    for i, axis_order in enumerate(axis_orders):
        print(i, axis_order)
        costs[i], coord_list = warp_path_cost(group_exists, start_coords, end_coords, axis_order)
        coord_history.append(coord_list)

    print(costs)

    index = np.argmin(costs)
    return axis_orders[index], costs[index], coord_history[index]


# =============================
# FLOW MODEL
# =============================
def solve_flow_field_1d(x, velocity_axis, flow_axes, radius, bidirectional=False, **kwargs):
    img_ndim = np.ndim(x)
    flow_ndim = len(flow_axes)

    # Convert negative indices
    def convert_index(ax_index):
        if ax_index < 0:
            return img_ndim + ax_index
        else:
            return ax_index
    flow_axes = tuple([convert_index(ax) for ax in flow_axes])
    # print(flow_axes)

    if len(radius) != img_ndim:
        raise ValueError('Radius must contain one entry for each dimension of x')

    size = tuple(2 * np.array(radius) + 1)

    # TODO: consider relaxing this constraint
    if size[velocity_axis] > 1:
        raise ValueError('radius along velocity_axis must be zero')

    if bidirectional:
        solver = bidirectional_flow
    else:
        solver = partial_flow_ilk

    # Axes with radius 0 (size 1): iterate over slices
    iter_axes = [ax for ax in range(img_ndim) if size[ax] == 1]
    stack_axes = [ax for ax in range(img_ndim) if size[ax] > 1]
    slice_radius = tuple([radius[ax] for ax in stack_axes])

    # if sigma is not None:
    #     if np.isscalar(sigma):
    #         slice_sigma = (sigma,) * len(stack_axes)
    #     else:
    #         slice_sigma = tuple([sigma[ax] for ax in stack_axes])
    # else:
    #     slice_sigma = None

    # Swap axes for iteration over slices
    x_swap = x.copy()
    swap_axes = iter_axes + stack_axes
    swap_v_axis = swap_axes.index(velocity_axis)
    slice_flow_axes = tuple([ax - len(iter_axes) for ax in flow_axes])
    # print(slice_flow_axes, slice_radius)
    for i, ax in enumerate(iter_axes[::-1]):
        x_swap = np.moveaxis(x_swap, ax + i, 0)

    # Initialize output array
    output = np.empty((*x_swap.shape, flow_ndim))
    output.fill(np.nan)

    distances = np.empty(x_swap.shape)
    distances.fill(np.nan)

    # Iterate over slices
    it = np.nditer(x_swap, op_axes=[list(range(len(iter_axes)))], flags=['multi_index'])
    for _ in it:
        ref_index = it.multi_index
        if ref_index[swap_v_axis] < x_swap.shape[swap_v_axis] - 1:
            x_ref = x_swap[ref_index]
            moving_index = list(ref_index).copy()
            solve = False
            distance = None
            x_moving = None
            if not np.all(np.isnan(x_ref)):
                # Find nearest neighbor slice along velocity_axis
                for i in range(ref_index[swap_v_axis] + 1, x_swap.shape[swap_v_axis]):
                    # print(i)
                    moving_index[swap_v_axis] = i
                    x_moving = x_swap[tuple(moving_index)]
                    if not np.all(np.isnan(x_moving)):
                        distance = i - ref_index[swap_v_axis]
                        solve = True
                        break

            # print(ref_index, moving_index, distance)

            if solve:
                # Assign zero weight to nans
                nan_mask = np.isnan(x_ref) | np.isnan(x_moving)
                weights = (~nan_mask).astype(float)
                # print(np.min(weights), np.max(weights))
                flow = solver(np.nan_to_num(x_ref), np.nan_to_num(x_moving),
                              flow_axes=slice_flow_axes, radius=slice_radius,
                              weights=weights,
                              **kwargs)
                # print(ref_index, moving_index, flow[slice_flow_axes].flatten()[:20])
                # print(output[ref_index].shape)
                output[ref_index] = np.moveaxis(flow, 0, flow.ndim - 1)[..., slice_flow_axes]
                distances[ref_index] = distance

    # Return axes to their original positions
    for i, ax in enumerate(iter_axes[::-1]):
        # Offset index by 1 for flow dim axis
        output = np.moveaxis(output, iter_axes.index(ax), ax)
        distances = np.moveaxis(distances, iter_axes.index(ax), ax)

    # Put flow dims first
    output = np.moveaxis(output, -1, 0)

    return output, distances


def solve_flow_field(x, velocity_axes, flow_axes, radii, bidirectional=False,
                     align=False, align_indices=None,
                     filter_flows=True, filter_func=None, filter_kw=None,
                     **kwargs):
    if align and align_indices is None:
        raise ValueError('align_indices must be provided if align=True')
    if align and len(align_indices) != len(velocity_axes):
        raise ValueError('Length of align_indices must match length of velocity_axes')

    flow_fields = []
    x_input = x.copy()
    for i, v_axis in enumerate(velocity_axes):
        flow, distance = solve_flow_field_1d(x_input, v_axis, flow_axes, radii[i],
                                             bidirectional=bidirectional,
                                             **kwargs)
        print(np.sum(~np.isnan(flow)))

        # Normalize flow to distance between slices
        flow = flow / np.expand_dims(distance, 0)

        if filter_flows:
            if filter_func is None:
                nan_mask = ~np.isnan(flow)
                flow = np.nan_to_num(flow)
                filter_func_i = partial(masked_filter, mask=nan_mask)
            else:
                filter_func_i = filter_func

            if filter_kw is None:
                filter_kw = {}

            flow = filter_func_i(flow, **filter_kw)

        print(np.sum(~np.isnan(flow)))

        flow_fields.append(flow)

        if align:
            # Align along current velocity_axis before proceeding to next axis
            x_input = align_to_reference_1d(x_input, flow, v_axis, flow_axes, align_indices[i])

    return flow_fields


def align_to_reference_1d(x, flow_field, velocity_axis, flow_axes, reference_index):
    x_align = np.empty_like(x)

    # Construct full flow from partial flow
    def build_full_flow(flow_in):
        flow_out = np.zeros((np.ndim(flow_in) - 1, *flow_in.shape[1:]))
        flow_out[flow_axes] = flow_in
        return flow_out

    # Mask nans for warping
    nan_mask = np.isnan(x)
    x = np.nan_to_num(x)

    # Place velocity axis first for indexing with write capability
    x_swap = np.moveaxis(x_align, velocity_axis, 0)

    for i in range(x.shape[velocity_axis]):
        warped = np.take(x, i, axis=velocity_axis).copy()

        # print(output_shape)

        if i == reference_index:
            # Original is reference - no warp
            pass
        if i <= reference_index:
            # Reverse flow
            flow_stack = np.take(flow_field, np.arange(i, reference_index), axis=velocity_axis + 1)
            # print(flow_stack.shape)

            flow_sequence = np.moveaxis(flow_stack, velocity_axis + 1, 0)
            # print(flow_sequence.shape)

            for flow in flow_sequence:
                flow = build_full_flow(flow)
                flow = reverse_flow(flow)
                # print(flow.shape, warped.shape)
                warped = warp(warped, flow)
        else:
            # Forward flow
            flow_stack = np.take(flow_field, np.arange(reference_index, i), axis=velocity_axis + 1)

            flow_sequence = np.moveaxis(flow_stack, velocity_axis + 1, 0)

            for flow in flow_sequence:
                flow = build_full_flow(flow)
                warped = warp(warped, flow)

        x_swap[i] = warped

    # Put nans back in place
    x_align[nan_mask] = np.nan

    return x_align


def align_to_reference(x, flow_field, velocity_axes, flow_axes, reference_indices):
    x_align = x.copy()

    for i, velocity_axis in enumerate(velocity_axes):
        x_align = align_to_reference_1d(x_align, flow_field[i],
                                        velocity_axis, flow_axes, reference_indices[i])

    return x_align

