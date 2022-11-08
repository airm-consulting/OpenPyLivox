import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("test.bin.pcd")
print(pcd)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.001,
                                  front=[-1, 0, 0],
                                  lookat=[0, 1, 0],
                                  up=[0, 0, 1])


tpcd = o3d.t.io.read_point_cloud("test.bin.pcd")
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
aabb.max_bound = (aabb.min_bound[0]*1.015625, aabb.max_bound[1], aabb.max_bound[2])
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
o3d.visualization.draw_geometries([downpcd])

# calc normal (optional for vivid visualization) 
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

print("showing noise removed points")
cl, denoised_ind = downpcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=1.0)
denoised_cloud = downpcd.select_by_index(denoised_ind)
noise_cloud = downpcd.select_by_index(denoised_ind, invert=True)
noise_cloud.paint_uniform_color([0, 0, 0])
o3d.visualization.draw_geometries([denoised_cloud, noise_cloud])

# fit plane 
pcd = denoised_cloud
plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# viz 
plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([1.0, 0, 0])

noneplane_cloud = pcd.select_by_index(inliers, invert=True)
noneplane_cloud.paint_uniform_color([0, 0, 1.0])

o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud])

hull, _ = plane_cloud.compute_convex_hull()
hull = hull.simplify_vertex_clustering(voxel_size=0.9)
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([hull_ls])
print(hull_ls)
print(np.asarray(hull.vertices))
#cluster_idx, cluster_no_of_triangle, area = hull.cluster_connected_triangles()
#print(cluster_no_of_triangle)
# o3d.visualization.draw_geometries([hull_normal], point_show_normal=True)

# np.asarray(tpcd.point.intensities)
# o3d.visualization.draw_geometries_with_vertex_selection([pcd])
