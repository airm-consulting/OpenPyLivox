import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("test.bin.pcd")
print(pcd)

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

# np.asarray(tpcd.point.intensities)
# o3d.visualization.draw_geometries_with_vertex_selection([pcd])

# o3d.visualization.draw_geometries([pcd],
#                                   zoom=10,
#                                   front=[-1, 0, 0],
#                                   lookat=[0, 1, 0],
#                                   up=[0, 0, 1])
# pcd.estimate_normals()
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# pcd.orient_normals_consistent_tangent_plane(1)
# o3d.visualization.draw([pcd])
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=5, num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                   zoom=1,
#                                   front=[-1, 0, 0],
#                                   lookat=[0, 1, 0],
#                                   up=[0, 0, 1])