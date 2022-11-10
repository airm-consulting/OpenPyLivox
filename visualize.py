import open3d as o3d
import numpy as np
import copy
import os
import numpy.linalg as LA
import itertools

filename = os.path.join("calibration", "6.pcd")
# filename = "test.bin.pcd"
pcd = o3d.io.read_point_cloud(filename)
print(pcd)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])


tpcd = o3d.t.io.read_point_cloud(filename)
intensities = tpcd.point['intensity'].numpy()
ind = np.where(intensities > 100)
#print(list(ind[0]))
inliers = pcd.select_by_index(ind[0].tolist())
outliers = pcd.select_by_index(ind[0].tolist(), invert=True)
outliers.paint_uniform_color([0.8, 0.8, 0.8])
inliers.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([inliers],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])

aabb = inliers.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = inliers.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([inliers, aabb, obb],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])
print(aabb)
extension_factor = (1.93)
aabb.max_bound = (aabb.min_bound[0]*extension_factor, aabb.max_bound[1], aabb.max_bound[2])
print(aabb)
cropped_pcd = inliers.crop(aabb)
o3d.visualization.draw_geometries([cropped_pcd, aabb],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])

print("Downsample the point cloud with a voxel")
#downpcd = cropped_pcd
downpcd = cropped_pcd.voxel_down_sample(voxel_size=0.0001)
o3d.visualization.draw_geometries([downpcd, aabb],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])
o3d.visualization.draw_geometries_with_vertex_selection([downpcd])

# # calc normal (optional for vivid visualization) 
# print("Recompute the normal of the downsampled point cloud")
# downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#     radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

print("showing noise removed points")
# denoised_cloud = downpcd
cl, denoised_ind = downpcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=1.0)
denoised_cloud = downpcd.select_by_index(denoised_ind)
noise_cloud = downpcd.select_by_index(denoised_ind, invert=True)
noise_cloud.paint_uniform_color([0, 0, 0])
o3d.visualization.draw_geometries([denoised_cloud, noise_cloud],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])

# fit plane 
def fit_plane(pc):
    thresh_list = [0.1, 0.02]
    for iter_time, thresh in enumerate(thresh_list):
        plane_model, plane_inliers = pc.segment_plane(distance_threshold=0.04,
                                                ransac_n=50,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Iter: {iter_time:d} Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        plane_cloud = pc.select_by_index(plane_inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])
        noneplane_cloud = pc.select_by_index(plane_inliers, invert=True)
        noneplane_cloud.paint_uniform_color([0, 0, 1.0])
        o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud],
                                        zoom=0.001,
                                        front=[-1, 0, 0],
                                        lookat=[0, 1, 0],
                                        up=[0, 0, 1])
        _pc = np.asarray(plane_cloud.points)
        x, y, z = _pc[:, 0], _pc[:, 1], _pc[:, 2]
        dist = np.abs(a * x + b * y + c * z + d) / LA.norm([a, b, c], ord=2)
        _pc = _pc[np.where(dist < thresh)]
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(_pc)
    return (output_pcd, a, b, c, d)

#project points to plane
def get_flattened_pcds2(pc,A,B,C,D,x0,y0,z0):
    source = copy.deepcopy(pc)
    x1 = np.asarray(source.points)[:,0]
    y1 = np.asarray(source.points)[:,1]
    z1 = np.asarray(source.points)[:,2]
    x0 = x0 * np.ones(x1.size)
    y0 = y0 * np.ones(y1.size)
    z0 = z0 * np.ones(z1.size)
    r = np.power(np.square(x1-x0)+np.square(y1-y0)+np.square(z1-z0),0.5)
    a = (x1-x0)/r
    b = (y1-y0)/r
    c = (z1-z0)/r
    t = -1 * (A * np.asarray(source.points)[:,0] + B * np.asarray(source.points)[:,1] + C * np.asarray(source.points)[:,2] + D)
    t = t / (a*A+b*B+c*C)
    np.asarray(source.points)[:,0] = x1 + a * t
    np.asarray(source.points)[:,1] = y1 + b * t
    np.asarray(source.points)[:,2] = z1 + c * t
    return source

fitted_pcd, A, B, C, D = fit_plane(denoised_cloud)
fitted_pcd = get_flattened_pcds2(fitted_pcd,A,B,C,D,0,0,0)
aabb = fitted_pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = fitted_pcd.get_oriented_bounding_box(robust=True)
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([fitted_pcd, aabb, obb],
                                  zoom=0.7,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])

#calculate distance between two points
def distance_between_two_points(p1,p2):
    return LA.norm(p1-p2)
#calculate shortest distance of points p from line formed by lp1 and lp2
# from https://stackoverflow.com/questions/50727961/shortest-distance-between-a-point-and-a-line-in-3-d-space
def distance_between_line_and_point(lp1, lp2, p):
    x = lp1-lp2
    return LA.norm(np.outer(np.dot(p-lp2, x)/np.dot(x, x), x)+lp2-p, axis=1)
def rectangle_perimeter_from_corners(corners):
    if not len(corners) == 4:
        print("error, must be 4 corners")
        return
    else:
        index_list = list(itertools.combinations(np.arange(0,4,dtype=int), 2))
        index_list = [list(index_list[i]) for i in range(len(index_list))] 
        boundaries_index_list = sorted(index_list, key= lambda x: distance_between_two_points(np.asarray(corners[x[0]]), np.asarray(corners[x[1]])))
        boundaries_index_list = boundaries_index_list[:4]
        boundaries_lines = []
        for x in range(len(boundaries_index_list)):
            boundaries_lines.append([corners[boundaries_index_list[x][0]], corners[boundaries_index_list[x][1]]])
        boundaries_lines = np.array(boundaries_lines)
        # print(boundaries_lines)
        return boundaries_lines, boundaries_index_list

def prepare_rectangle_perimeter_lineset(corners, boundaries_index_list, rgb):
    colors = [rgb for i in range(len(boundaries_index_list))]
    line_set = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(corners), 
                                    lines = o3d.utility.Vector2iVector(boundaries_index_list),)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

#find points that is the cloest to each boundary line of the box
corners_points = np.unique(np.asarray(obb.get_box_points()), axis=0)
corners = [list(corners_points[0]), list(corners_points[2]), list(corners_points[4]), list(corners_points[6]),]
print(corners)
boundaries_lines, boundaries_index_list = rectangle_perimeter_from_corners(corners)
#draw boundaries
print(boundaries_index_list)
line_set = prepare_rectangle_perimeter_lineset(corners=corners, boundaries_index_list=boundaries_index_list, rgb=[1,0,0])
o3d.visualization.draw_geometries([fitted_pcd, line_set])
corner_point_idx = []
for i in range(len(boundaries_lines)):
    d = distance_between_line_and_point(boundaries_lines[i][0], boundaries_lines[i][1], fitted_pcd.points)
    corner_index_list = np.where(d <= (0.005))[0].tolist()
    for idx in corner_index_list:
        corner_point_idx.append(idx)
print(corner_point_idx)
corner_points = fitted_pcd.select_by_index(corner_point_idx)
corner_points.paint_uniform_color([0,0,0])
# _, b_idx = rectangle_perimeter_from_corners(corner_points.points)
# corner_lines = prepare_rectangle_perimeter_lineset(corner_points.points, b_idx, [1,0,0])
inliers.paint_uniform_color([0.8,0.8,0.8])
o3d.visualization.draw_geometries([inliers, corner_points, obb],
                                  zoom=0.0001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])
print(np.asarray(corner_points.points))


# plane_cloud_deep_copy = copy.deepcopy(plane_cloud)
# pcd_deep_copy = copy.deepcopy(pcd)
# plane_cloud_project = get_flattened_pcds2(pcd_deep_copy, a, b, c, d, 0, 0, 0)
# plane_cloud_project.paint_uniform_color([0,0,1])
# o3d.visualization.draw_geometries([plane_cloud, plane_cloud_project])

# hull, _ = plane_cloud.compute_convex_hull()
# hull = hull.simplify_vertex_clustering(voxel_size=0.1)
# hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
# hull_ls.paint_uniform_color((1, 0, 0))
# plane_cloud.paint_uniform_color([0.8, 0.8, 0.8])
# o3d.visualization.draw_geometries([hull_ls, plane_cloud])
# print(hull_ls)
# print(np.asarray(hull.vertices))

#cluster_idx, cluster_no_of_triangle, area = hull.cluster_connected_triangles()
#print(cluster_no_of_triangle)
# o3d.visualization.draw_geometries([hull_normal], point_show_normal=True)

# np.asarray(tpcd.point.intensities)
# o3d.visualization.draw_geometries_with_vertex_selection([pcd])
  