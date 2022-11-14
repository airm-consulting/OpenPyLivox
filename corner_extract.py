import open3d as o3d
import numpy as np
import copy
import os
from os import path
import numpy.linalg as LA
import itertools
import sys, getopt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import transforms3d
import matplotlib.path as mplPath

DEBUG_VISUALIZE = True
WIDTH = 0.455 #in m
LENGTH = 0.600 #in m

def load_pointcloud(filename):
    tpcd = o3d.t.io.read_point_cloud(filename)
    point_np = tpcd.point['positions'].numpy()
    intensities = tpcd.point['intensity'].numpy()
    pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_np))
    return pc, intensities

def filter_by_intensity(pc, intensities, intensity_thres):
    if DEBUG_VISUALIZE:
        o3d.visualization.draw_geometries([pc],
                                        zoom=0.001,
                                        front=[-1, 0, 0],
                                        lookat=[0, 1, 0],
                                        up=[0, 0, 1])
    ind = np.where(intensities > intensity_thres)
    inliers = pc.select_by_index(ind[0].tolist())
    outliers = pc.select_by_index(ind[0].tolist(), invert=True)
    outliers.paint_uniform_color([0.8, 0.8, 0.8])
    inliers.paint_uniform_color([1, 0, 0])
    if DEBUG_VISUALIZE:
        o3d.visualization.draw_geometries([inliers],
                                        zoom=0.001,
                                        front=[-1, 0, 0],
                                        lookat=[0, 1, 0],
                                        up=[0, 0, 1])
    return inliers, outliers

def filter_by_bound(pc, extension_factor):
    aabb = pc.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = pc.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    if DEBUG_VISUALIZE:
        o3d.visualization.draw_geometries([pc, aabb, obb],
                                        zoom=0.001,
                                        front=[-1, 0, 0],
                                        lookat=[0, 1, 0],
                                        up=[0, 0, 1])
    print(aabb)
    aabb.max_bound = (aabb.min_bound[0]*extension_factor, aabb.max_bound[1], aabb.max_bound[2])
    print(aabb)
    cropped_pcd = pc.crop(aabb)
    if DEBUG_VISUALIZE:
        o3d.visualization.draw_geometries([cropped_pcd, aabb],
                                        zoom=0.001,
                                        front=[-1, 0, 0],
                                        lookat=[0, 1, 0],
                                        up=[0, 0, 1])
    return cropped_pcd

def downsample_by_voxel(pc, voxel_size):
    print("Downsample the point cloud with a voxel")
    downpcd = pc.voxel_down_sample(voxel_size=voxel_size)
    if DEBUG_VISUALIZE:
        o3d.visualization.draw_geometries([downpcd])
    # o3d.visualization.draw_geometries_with_vertex_selection([downpcd])
    return downpcd

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html#Statistical-outlier-removal
def remove_outliers(pc, neighbor_cnt, std_ratio):
    print("showing noise removed points")
    cl, denoised_ind = pc.remove_statistical_outlier(nb_neighbors=neighbor_cnt, std_ratio=std_ratio)
    denoised_cloud = pc.select_by_index(denoised_ind)
    if DEBUG_VISUALIZE:
        display_inlier_outlier(pc, denoised_ind)
    return denoised_cloud

def remove_circular_outliers(pc, neighbor_cnt, radius):
    print("Radius oulier removal")
    cl, ind = pc.remove_radius_outlier(nb_points=neighbor_cnt, radius=radius)
    denoised_cloud = pc.select_by_index(ind)
    if DEBUG_VISUALIZE:
        display_inlier_outlier(pc, ind)
    return denoised_cloud

# fit plane 
def fit_plane(pc):
    thresh_list = [0.1, 0.02]
    output_pc=pc
    for iter_time, thresh in enumerate(thresh_list):
        plane_model, plane_inliers = output_pc.segment_plane(distance_threshold=0.04,
                                                ransac_n=50,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Iter: {iter_time:d} Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        plane_cloud = pc.select_by_index(plane_inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])
        noneplane_cloud = pc.select_by_index(plane_inliers, invert=True)
        noneplane_cloud.paint_uniform_color([0, 0, 1.0])
        if DEBUG_VISUALIZE:
            o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud])
        _pc = np.asarray(plane_cloud.points)
        x, y, z = _pc[:, 0], _pc[:, 1], _pc[:, 2]
        dist = np.abs(a * x + b * y + c * z + d) / LA.norm([a, b, c], ord=2)
        _pc = _pc[np.where(dist < thresh)]
        output_pc = o3d.geometry.PointCloud()
        output_pc.points = o3d.utility.Vector3dVector(_pc)
    return (output_pc, a, b, c, d)

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
    if DEBUG_VISUALIZE:
        o3d.visualization.draw_geometries([source])
    return source

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

def transfer_by_pca(pc):
    """
    transfer chessboard to xOy plane
    REF: https://github.com/mfxox/ILCC/tree/master/ILCC
    :param pc:
    :return:
    """
    # to rotate the arr correctly, the direction of the axis has to be found correct(PCA also shows the axis but not the direction of axis)
    #
    pca = PCA(n_components=3)
    pca.fit(pc)

    ####################################################
    # there are C(6,3) possible axis combination, so the following constraint should be applied:
    # there are there requirements for the coordinates axes for the coordinate system of the chessboard
    # 1. right-hand rule
    # 2. z axis should point to the origin
    # 3. the angle between y axis of chessboard and z axis of LiDAR point cloud less than 90 deg
    ####################################################
    trans_mat = pca.components_
    # switch x and y axis
    trans_mat[[0, 1]] = trans_mat[[1, 0]]
    # cal z axis to obey the right hands
    trans_mat[2] = np.cross(trans_mat[0], trans_mat[1])

    # to make angle between y axis of chessboard and z axis of LiDAR point cloud less than 90 deg
    sign2 = np.sign(np.dot(trans_mat[1], np.array([0, 0, 1])))
    # print "sign2", sign2
    # we only need the property that Y-axis to be transformed, but x-axis must be transformed together to keep right-hand property
    trans_mat[[0, 1]] = sign2 * trans_mat[[0, 1]]

    # to make the norm vector point to the side where the origin exists
    # the angle  between z axis and the vector  from one point on the board to the origin should  be less than 90 deg
    sign = np.sign(np.dot(trans_mat[2], 0 - pc.mean(axis=0).T))

    # print "sign", sign
    # need Z-axis to be transformed, we transform X-axis along with it just to keep previous property
    trans_mat[[0, 2]] = sign * trans_mat[[0, 2]]

    transfered_pc = np.dot(pc, trans_mat.T)
    # print pca.components_
    # print "x,y,cross", np.cross(pca.components_[1], pca.components_[2])

    return trans_mat, transfered_pc

def transform_to_xoy_plane(pc):
    rot1, transed_pcd = transfer_by_pca(np.asarray(pc.points))
    t1 = transed_pcd.mean(axis=0)
    transed_pcd = transed_pcd - t1
    transed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(transed_pcd))
    if DEBUG_VISUALIZE:
        transed_pc.paint_uniform_color([1,0,0])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pc, transed_pc, frame])
    return transed_pc, rot1, t1

def objective_for_pca(theta_t, transed_pcd, l, w):
    transed_pcd_for_costf = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], theta_t[0]),
								(transed_pcd + np.array([[theta_t[1], theta_t[2], 0]])).T).T
    bound = np.array([[-w/2,-l/2],[-w/2,l/2], [w/2,l/2], [w/2,-l/2]])
    polygon_path = mplPath.Path(bound)
    p2d = transed_pcd_for_costf[:, :2]
    in_polygon = polygon_path.contains_points(p2d)
    pts_not_in_polygon = transed_pcd_for_costf[~in_polygon]
    return np.shape(pts_not_in_polygon)[0]

def fit_rectangle(pc, length, width):
    res = minimize(objective_for_pca, np.zeros(3).tolist(), args=(np.asarray(pc.points),length,width), method='Powell', tol=1e-10, options={"maxiter":10000000})
    if DEBUG_VISUALIZE:
        trans_transed_pcd = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], res.x[0]),
							(np.asarray(pc.points) + np.array([[res.x[1], res.x[2], 0]])).T).T
        trans_transed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(trans_transed_pcd))	
        trans_transed_pc.paint_uniform_color([0,0,1])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([trans_transed_pc, pc, frame,],)	
    return res.x

def obtain_corners(pc, length, width, original_pc=None):
    downpc = downsample_by_voxel(pc, 0.001)
    transed_pc, rot1, t1 = transform_to_xoy_plane(downpc)
    rt2 = fit_rectangle(transed_pc, length, width)
    imaginary_box_corners = np.array([[-width/2,-length/2,0],[-width/2,length/2,0], [width/2,length/2,0], [width/2,-length/2,0]]) 
    imaginary_box_corners[:, :3] = np.dot(imaginary_box_corners[:, :3], np.linalg.inv(transforms3d.axangles.axangle2mat([0, 0, 1], rt2[0]).T)) - np.array([rt2[1], rt2[2], 0])
    imaginary_box_corners[:, :3] = np.dot(imaginary_box_corners[:, :3] + t1, np.linalg.inv(rot1.T))
    if DEBUG_VISUALIZE:
        imaginary_box_line, imaginary_box_line_idx_list = rectangle_perimeter_from_corners(imaginary_box_corners)
        line_set = prepare_rectangle_perimeter_lineset(imaginary_box_corners, imaginary_box_line_idx_list, [1,0,0])
        imaginary_box_corners_pc = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(imaginary_box_corners))
        imaginary_box_corners_pc.paint_uniform_color([1,0,0])
        if original_pc is None:
            original_pc = copy.deepcopy(pc)
        original_pc.paint_uniform_color([0.8,0.8,0.8])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([line_set, imaginary_box_corners_pc, original_pc, frame,],
			# front = np.array([ 0.92634022122237125, 0.33279144978698516, -0.17647562294652724 ]),
			# lookat = np.array([ 5.0959061939605039, -0.95655826980403702, 0.73581584662371835 ]),
			# up = np.array([ 0.15983800796975584, 0.076961395021687196, 0.98413858520260056 ]),
			# zoom = 0.02
        )
    return imaginary_box_corners
# #find points that is the cloest to each boundary line of the box
# corners_points = np.unique(np.asarray(obb.get_box_points()), axis=0)
# corners = [list(corners_points[0]), list(corners_points[2]), list(corners_points[4]), list(corners_points[6]),]
# print(corners)
# boundaries_lines, boundaries_index_list = rectangle_perimeter_from_corners(corners)
# #draw boundaries
# print(boundaries_index_list)
# line_set = prepare_rectangle_perimeter_lineset(corners=corners, boundaries_index_list=boundaries_index_list, rgb=[1,0,0])
# o3d.visualization.draw_geometries([fitted_pcd, line_set])
# corner_point_idx = []
# for i in range(len(boundaries_lines)):
#     d = distance_between_line_and_point(boundaries_lines[i][0], boundaries_lines[i][1], fitted_pcd.points)
#     corner_index_list = np.where(d <= (0.005))[0].tolist()
#     for idx in corner_index_list:
#         corner_point_idx.append(idx)
# print(corner_point_idx)
# corner_points = fitted_pcd.select_by_index(corner_point_idx)
# corner_points.paint_uniform_color([0,0,0])
# # _, b_idx = rectangle_perimeter_from_corners(corner_points.points)
# # corner_lines = prepare_rectangle_perimeter_lineset(corner_points.points, b_idx, [1,0,0])
# inliers.paint_uniform_color([0.8,0.8,0.8])
# o3d.visualization.draw_geometries([inliers, corner_points, obb],
#                                   zoom=0.0001,
#                                   front=[-1, 0, 0],
#                                   lookat=[0, 1, 0],
#                                   up=[0, 0, 1])
# print(np.asarray(corner_points.points))

def printHelp():
    print('Usage: visualize.py -f <pcd_file>')

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:f:", ["help", "file="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    if len(opts) < 1:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-f", "--file"):
            # folder_separator='/'
            # if (os.name == 'nt'):
            #     folder_separator='\\'
            # elif (os.name == 'posix'):
            #     folder_separator='/'    
            # cwd = path.abspath(__file__).rsplit(folder_separator, 1)[0] + folder_separator
            # filename = arg.rsplit(folder_separator, 1)[-1].split('.')[0]
            points, intensities = load_pointcloud(arg)
            points_with_high_intensity,_ = filter_by_intensity(points, intensities, 100)
            points_within_bound = filter_by_bound(points_with_high_intensity, 1.93)
            clean_points = remove_outliers(points_within_bound, 6, 1.0)
            clean_points = remove_circular_outliers(clean_points, 16, 0.01)
            fitted_pcd, A, B, C, D = fit_plane(clean_points)
            fitted_pcd = get_flattened_pcds2(fitted_pcd,A,B,C,D,0,0,0)
            corners = obtain_corners(fitted_pcd, LENGTH, WIDTH, original_pc=clean_points)
            print(corners)
        else:
            printHelp()
            sys.exit(2)