import numpy as np
import kmapper as kmap
import networkx as nx


def image_to_cloud(img, dim_grids, thresh=None, index=None, include_intensity=True, return_index=False):
    if len(dim_grids) != np.ndim(img):
        raise ValueError('dim_grids must match image dimensions')

    if index is None and thresh is None:
        raise ValueError('Either thresh or index must be provided')

    coord_mesh = np.meshgrid(*dim_grids, indexing='ij')

    if index is None:
        index = img > thresh

    values = [cm[index] for cm in coord_mesh]
    if include_intensity:
        values.append(img[index])

    cloud = np.stack(values, axis=0).T

    if return_index:
        return cloud, index
    else:
        return cloud


def cloud_to_image(cloud, index, fill_val=0):
    img = np.empty(index.shape)
    img.fill(fill_val)

    img[index] = cloud

    return img


def component_members(graph, component_nodes):
    members = [graph['nodes'][node] for node in component_nodes]
    return np.unique(np.concatenate(members))


def connected_component_members(graph, nx_graph=None):
    if nx_graph is None:
        nx_graph = kmap.adapter.to_nx(graph)
    components = list(nx.connected_components(nx_graph))

    return [component_members(graph, nodes) for nodes in components]