import open3d as o3d
import numpy as np
import copy
import os

filename = os.path.join("calibration", "5.pcd")
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
downpcd = cropped_pcd
# downpcd = cropped_pcd.voxel_down_sample(voxel_size=0.05)
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
fitted_pcd = denoised_cloud
plane_model, plane_inliers = denoised_cloud.segment_plane(distance_threshold=0.005,
                                         ransac_n=10,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# viz 
plane_cloud = denoised_cloud.select_by_index(plane_inliers)
plane_cloud.paint_uniform_color([1.0, 0, 0])
noneplane_cloud = denoised_cloud.select_by_index(plane_inliers, invert=True)
noneplane_cloud.paint_uniform_color([0, 0, 1.0])
o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])

aabb = plane_cloud.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = plane_cloud.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([plane_cloud, aabb, obb],
                                  zoom=0.7,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])
print(np.asarray(aabb.get_box_points()))

inliers.paint_uniform_color([0.8,0.8,0.8])
o3d.visualization.draw_geometries([inliers, aabb, obb],
                                  zoom=0.0001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])

# def get_flattened_pcds2(source,A,B,C,D,x0,y0,z0):
#     x1 = np.asarray(source.points)[:,0]
#     y1 = np.asarray(source.points)[:,1]
#     z1 = np.asarray(source.points)[:,2]
#     x0 = x0 * np.ones(x1.size)
#     y0 = y0 * np.ones(y1.size)
#     z0 = z0 * np.ones(z1.size)
#     r = np.power(np.square(x1-x0)+np.square(y1-y0)+np.square(z1-z0),0.5)
#     a = (x1-x0)/r
#     b = (y1-y0)/r
#     c = (z1-z0)/r
#     t = -1 * (A * np.asarray(source.points)[:,0] + B * np.asarray(source.points)[:,1] + C * np.asarray(source.points)[:,2] + D)
#     t = t / (a*A+b*B+c*C)
#     np.asarray(source.points)[:,0] = x1 + a * t
#     np.asarray(source.points)[:,1] = y1 + b * t
#     np.asarray(source.points)[:,2] = z1 + c * t
#     return source
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
  