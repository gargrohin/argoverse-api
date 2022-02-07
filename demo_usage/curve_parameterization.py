import os
# from skimage import io, transform
import math
import numpy as np
import pandas as pd
import cv2
import yaml
# import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
# import imutils
import glob
import json
import matplotlib.pyplot as plt

# from torchvision import transforms, utils

def get_yaw(quat, use_degrees=True):
    """
    Description: Given a quaternion: np.array([x,y,z,w]), return the its corresding yaw.
    Return: angle (degrees)
    """
    r = R.from_quat(quat)
    rot_vec = r.as_euler('zyx', degrees=use_degrees)
    
    return rot_vec[0]

def visualize_graph(i, traversed_traj, planned_traj):
    # stop signs (2)
    t_stop_indx = np.where(traversed_traj[2, :] == 2)
    p_stop_indx = np.where(planned_traj[2, :] == 2)

    # crossings (3)
    t_crossing_indx = np.where(traversed_traj[2, :] == 3)
    p_crossing_indx = np.where(planned_traj[2, :] == 3)

    # generate plots
    plt.figure()
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    plt.scatter(traversed_traj[0, :], traversed_traj[1, :], s=2, marker='o')
    plt.scatter(planned_traj[0, :], planned_traj[1, :], s=2, marker='s')

    # plot stopsigns/signals
    plt.scatter(traversed_traj[0, t_stop_indx], 
                traversed_traj[1, t_stop_indx], marker='*')
    plt.scatter(planned_traj[0, p_stop_indx], 
                planned_traj[1, p_stop_indx], marker='*')
    
    # plot crossings
    plt.scatter(traversed_traj[0, t_crossing_indx], 
                traversed_traj[1, t_crossing_indx], marker='X')
    plt.scatter(planned_traj[0, p_crossing_indx], 
                planned_traj[1, p_crossing_indx], marker='X')

    
    plt.savefig('osm_graph_viz/' + str(i) + '.png')
    plt.close()

def get_osm_trajectories(data_segment, max_traversed, max_planned):

    road_network = data_segment['road_network']
    traversed_path = data_segment['traversed_path']
    planned_path = data_segment['planned_path']
    stops = data_segment['stops']
    crossings = data_segment['crossings']
    signals = data_segment['signals']

    traversed_len = len(traversed_path)
    planned_len = len(planned_path)
    
    traversed_np = np.zeros((3, max_traversed))
    planned_np = np.zeros((3, max_planned))

    #traversed_np[2, :] = 0
    #planned_np[2, :] = 1 
    # nodes part of the traversed trajectory
    # valid elements: lat, lon, adjacency list, 2D rel-coordinates

    for i in range(min(max_traversed, traversed_len)):
        l_index = traversed_len - i - 1

        if str(traversed_path[l_index]) not in road_network['nodes'].keys():
            continue

        #node_i = road_network['nodes'][str(traversed_path[l_range - i - 1])]
        node_i = road_network['nodes'][str(traversed_path[l_index])]

        # (x, y) coordinates
        traversed_np[0, i] = node_i['pose'][0] 
        traversed_np[1, i] = node_i['pose'][1] 

        # feature type: planned(0), traversed(1), stop/signal(2), crossing(3)
        if traversed_path[l_index] in stops or traversed_path[l_index] in signals:
            traversed_np[2, i] = 2 
        #elif traversed_path[l_index] in crossings:
        #    traversed_np[2, i] = 3 

    # reverse traversed_np 
    traversed_np = np.flip(traversed_np, 1)
    
    # nodes part of the future trajectory
    for i in range(min(max_planned, planned_len)):
        if str(planned_path[i]) not in road_network['nodes'].keys():
            continue
        node_i = road_network['nodes'][str(planned_path[i])]
        # (x, y) coordinates
        planned_np[0, i] = node_i['pose'][0] 
        planned_np[1, i] = node_i['pose'][1] 
    
        # feature type: planned(0), traversed(1), stop/signal(2), crossing(3)
        if planned_path[i] in stops or planned_path[i] in signals:
            planned_np[2, i] = 2 
        #elif planned_path[i] in crossings:
        #    planned_np[2, i] = 3 


    return traversed_np, planned_np
  
    
def get_interpolated_ego_trajectories_json(data_path, cfg, save_visualization=False, semantic_tf=None, use_gpu=True):

    #x_origin = cfg["maps"]["x_origin"]
    #y_origin = cfg["maps"]["y_origin"]
    resolution = cfg["maps"]["resolution"]
    waypoint_horizon = cfg["nn_params"]["horizon"]
    inbetween_thresh = cfg["interpolation_thresholds"]["inbetween_distance"]
    density = cfg["interpolation_thresholds"]["density"]
    speed_thresh = cfg["interpolation_thresholds"]["speed_threshold"]
    distance_horizon = (waypoint_horizon * inbetween_thresh) 
    
    loaded_data = []

    if save_visualization:
        os.mkdir("lbpsm_viz")
        os.mkdir("osm_graph_viz")


    curr_json = None

    # Read lbpsm data between range_start and range_end and extract feasible trajectories
    for data_segment in glob.iglob(data_path + '/**'):

        #json_files = data_path + "graphs/"
        json_files = data_segment + "/graphs/"
        num_samples = len([name for name in os.listdir(json_files) if os.path.isfile(os.path.join(json_files, name))])


        # process json file
        poses = np.zeros((num_samples, 7)) 
        speed = np.zeros(num_samples)

        for j in range(num_samples):
            with open(json_files + str(j) + '.json') as f:
                curr_json = json.load(f)
                
            poses[j, :] = np.asarray(curr_json['pose']) 
            speed[j] = curr_json['speed']
        
             
            
        for j in range(poses.shape[0]):
            with open(json_files + str(j) + '.json') as f:
                curr_json = json.load(f)
                

            # retrieve traversed/planned trajectories from OSM representation
            traversed_traj, planned_traj = get_osm_trajectories(curr_json, 20, 20)
            if save_visualization:
                visualize_graph(j, traversed_traj, planned_traj)

            # being interpolation process
            waypoints = poses[j:]
            current_speed = speed[j:]
            current_distance = 0
            prev_waypoint = 0

            ego_vehicle_pose = waypoints[0,0:2]
            yaw_ego = get_yaw(waypoints[0, 3:], False)

            interpolated_wp = np.hstack((ego_vehicle_pose, yaw_ego)).reshape((1,3))
            N = waypoints.shape[0]
            
            # If ego-vehicle is moving relatively slow, continue
            if current_speed[0] <= speed_thresh:
                continue

            # Collect enough waypoints to cover 'distance_horizon' meters and
            # interpolate
            for k in range(1, N, 1):
                distance = waypoints[k, 0:2] - waypoints[prev_waypoint, 0:2]
                curr_speed = current_speed[k]
                dx = np.sqrt(np.sum(distance ** 2))
                
                if dx <= density:
                    continue

                current_distance += dx

                # Interpolate
                x0 = waypoints[prev_waypoint, 0]
                y0 = waypoints[prev_waypoint, 1]
                yaw0 = get_yaw(waypoints[prev_waypoint, 3:], False)

                xf = waypoints[k, 0]
                yf = waypoints[k, 1]
                yawf = get_yaw(waypoints[k, 3:], False)

                N_s = int(dx/density)
                inter_nodes = []
                for s in range(N_s):
                    x_s = ((N_s - s)*x0 + (s + 1)*xf)/(N_s + 1)
                    y_s = ((N_s - s)*y0 + (s + 1)*yf)/(N_s + 1)
                    yaw_s = ((N_s - s)*yaw0 + (s + 1)*yawf)/(N_s + 1)
                    inter_nodes.append(np.hstack((x_s, y_s, yaw_s)).reshape((1,3)))

                inter_nodes.append(np.hstack((xf, yf, yawf)).reshape((1,3)))
                inter_nodes = np.vstack(inter_nodes)    
                interpolated_wp = np.vstack((interpolated_wp, inter_nodes))

                prev_waypoint = k

                if current_distance >= 2*distance_horizon:
                    break

            # Retrive final set of waypoints spaced 'inbetween_thresh' apart 
            curr_inbetween = 0
            prev_waypoint = 0
            final_wp = []
            final_wp.append(interpolated_wp[0, 0:3].reshape((1,3)))

            for k in range(1, interpolated_wp.shape[0], 1):
                distance = interpolated_wp[k, 0:2] - interpolated_wp[prev_waypoint, 0:2]
                dx = np.sqrt(np.sum(distance ** 2))
                #print("dx: " + str(dx))
                
                #curr_inbetween += dx
                if dx >= inbetween_thresh:
                    #print("curr_inbetween: " + str(curr_inbetween))
                    final_wp.append(interpolated_wp[k, 0:3].reshape((1,3)))
                    prev_waypoint = k
                    curr_inbetween = 0

                if len(final_wp) == waypoint_horizon+1:
                    break
                
            final_wp = np.vstack(final_wp)
                     
            waypoints = final_wp 
            N = waypoints.shape[0]
            waypoints_px_u = []
            waypoints_px_v = []

            waypoints_x = []
            waypoints_y = []
            ego_vehicle_pose = []
            yaw_ego = 0

            if N != waypoint_horizon+1:
                continue
            
            # Transform waypoints into ego-vehicle frame and pixel coordinates
            for k in range(N):
                if k == 0:
                    ego_vehicle_pose = waypoints[0,0:2]
                    yaw_ego = waypoints[0, 2]
                    continue

                curr_yaw = waypoints[k, 2]
                curr_position = waypoints[k, 0:2]

                # make waypoints wrt ego vehicle 
                rel_pose = curr_position - ego_vehicle_pose
                rel_pose = rel_pose.reshape((2, 1))
                rel_pose_pix = np.copy(rel_pose)
                rel_pose_pix[0] = int(rel_pose_pix[0] / resolution)
                rel_pose_pix[1] = int(rel_pose_pix[1] / resolution)
                
                # Determine rotation needed for waypoint
                theta = np.pi - yaw_ego + (3*np.pi/180)

                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                # XY/px positions
                rot_pose_pix = np.matmul(R, rel_pose_pix)
                rot_pose_xy = np.matmul(R, rel_pose)

                rot_pose_x = rot_pose_xy[0]
                rot_pose_y = rot_pose_xy[1]
                rot_pose_u = rot_pose_pix[0]
                rot_pose_v = rot_pose_pix[1]

                # Waypoints XY
                waypoints_x.append(rot_pose_x)
                waypoints_y.append(rot_pose_y)

                # Waypoints px
                waypoints_px_u.append(rot_pose_u)
                waypoints_px_v.append(rot_pose_v)


            waypoints_x = np.asarray(waypoints_x)
            waypoints_y = np.asarray(waypoints_y)

            waypoints_px_u = np.asarray(waypoints_px_u)
            waypoints_px_v = np.asarray(waypoints_px_v)
            
            waypoints_xy = np.hstack((waypoints_x, waypoints_y)).T
            waypoints_px = np.hstack((waypoints_px_u, waypoints_px_v)).T

            # Load semantic bev 
            lbpsm_map = cv2.imread(data_segment + "/semantic/" + str(j) + ".png")
            lbpsm_cp = np.copy(lbpsm_map)


            if semantic_tf:
                lbpsm_map = semantic_tf(lbpsm_map)
                if use_gpu and torch.cuda.is_available():
                    lbpsm_map = lbpsm_map.to("cuda")

            waypoints_px_tf = torch.from_numpy(waypoints_px)
            waypoints_xy_tf = torch.from_numpy(waypoints_xy)
            traversed_tf = torch.from_numpy(traversed_traj.copy())
            planned_tf = torch.from_numpy(planned_traj.copy())

            if use_gpu and torch.cuda.is_available():
                waypoints_px_tf = waypoints_px_tf.float().cuda()                
                waypoints_xy_tf = waypoints_xy_tf.float().cuda()                
                traversed_tf = traversed_tf.float().cuda()                
                planned_tf = planned_tf.float().cuda()                

            loaded_data.append([j, data_segment, lbpsm_map, waypoints_px_tf, waypoints_xy_tf, 
                                traversed_tf, planned_tf])
            #loaded_data.append([j, data_segment, lbpsm_map, waypoints_px, waypoints_xy])

            # Save visualization
            if save_visualization:
                
                for k in range(waypoints_px.shape[1]):
                    lbpsm_cp = cv2.circle(lbpsm_cp, 
                            (int(waypoints_px[1, k]+cfg["nn_params"]["ego_center_v"]), 
                             int(waypoints_px[0, k]+cfg["nn_params"]["ego_center_u"])),
                            1, (255,0,0), -1)
                cv2.imwrite("lbpsm_viz/" + str(j) + ".png", lbpsm_cp)
            # Assert the number of waypoints generate equals horizon
    
    print("Data summary:")
    print("Number of samples: " + str(len(loaded_data)))
    return loaded_data


def get_interpolated_ego_trajectories(data_path, cfg, save_visualization=False, semantic_tf=None, osm_tf=None, use_gpu=True):

    #x_origin = cfg["maps"]["x_origin"]
    #y_origin = cfg["maps"]["y_origin"]
    resolution = cfg["maps"]["resolution"]
    waypoint_horizon = cfg["nn_params"]["horizon"]
    inbetween_thresh = cfg["interpolation_thresholds"]["inbetween_distance"]
    density = cfg["interpolation_thresholds"]["density"]
    speed_thresh = cfg["interpolation_thresholds"]["speed_threshold"]
    distance_horizon = (waypoint_horizon * inbetween_thresh) 
    
    loaded_data = []

    if save_visualization:
        os.mkdir("lbpsm_viz")


    # Read lbpsm data between range_start and range_end and extract feasible trajectories
    for data_segment in glob.iglob(data_path + '/**'):
        poses = np.load(data_segment + "/poses.npy")
        speed = np.load(data_segment + "/speed.npy")

        for j in range(poses.shape[0]):
            waypoints = poses[j:]
            current_speed = speed[j:]
            current_distance = 0
            prev_waypoint = 0

            ego_vehicle_pose = waypoints[0,0:2]
            yaw_ego = get_yaw(waypoints[0, 3:], False)

            interpolated_wp = np.hstack((ego_vehicle_pose, yaw_ego)).reshape((1,3))
            N = waypoints.shape[0]
            
            # If ego-vehicle is moving relatively slow, continue
            if current_speed[0] <= speed_thresh:
                continue

            # Collect enough waypoints to cover 'distance_horizon' meters and
            # interpolate
            for k in range(1, N, 1):
                distance = waypoints[k, 0:2] - waypoints[prev_waypoint, 0:2]
                curr_speed = current_speed[k]
                dx = np.sqrt(np.sum(distance ** 2))
                
                if dx <= density:
                    continue

                current_distance += dx

                # Interpolate
                x0 = waypoints[prev_waypoint, 0]
                y0 = waypoints[prev_waypoint, 1]
                yaw0 = get_yaw(waypoints[prev_waypoint, 3:], False)

                xf = waypoints[k, 0]
                yf = waypoints[k, 1]
                yawf = get_yaw(waypoints[k, 3:], False)

                N_s = int(dx/density)
                inter_nodes = []
                for s in range(N_s):
                    x_s = ((N_s - s)*x0 + (s + 1)*xf)/(N_s + 1)
                    y_s = ((N_s - s)*y0 + (s + 1)*yf)/(N_s + 1)
                    yaw_s = ((N_s - s)*yaw0 + (s + 1)*yawf)/(N_s + 1)
                    inter_nodes.append(np.hstack((x_s, y_s, yaw_s)).reshape((1,3)))

                inter_nodes.append(np.hstack((xf, yf, yawf)).reshape((1,3)))
                inter_nodes = np.vstack(inter_nodes)    
                interpolated_wp = np.vstack((interpolated_wp, inter_nodes))

                prev_waypoint = k

                if current_distance >= 2*distance_horizon:
                    break

            # Retrive final set of waypoints spaced 'inbetween_thresh' apart 
            curr_inbetween = 0
            prev_waypoint = 0
            final_wp = []
            final_wp.append(interpolated_wp[0, 0:3].reshape((1,3)))

            for k in range(1, interpolated_wp.shape[0], 1):
                distance = interpolated_wp[k, 0:2] - interpolated_wp[prev_waypoint, 0:2]
                dx = np.sqrt(np.sum(distance ** 2))
                #print("dx: " + str(dx))
                
                #curr_inbetween += dx
                if dx >= inbetween_thresh:
                    #print("curr_inbetween: " + str(curr_inbetween))
                    final_wp.append(interpolated_wp[k, 0:3].reshape((1,3)))
                    prev_waypoint = k
                    curr_inbetween = 0

                if len(final_wp) == waypoint_horizon+1:
                    break
                
            final_wp = np.vstack(final_wp)
                     
            waypoints = final_wp 
            N = waypoints.shape[0]
            waypoints_px_u = []
            waypoints_px_v = []

            waypoints_x = []
            waypoints_y = []
            ego_vehicle_pose = []
            yaw_ego = 0

            if N != waypoint_horizon+1:
                continue
            
            # Transform waypoints into ego-vehicle frame and pixel coordinates
            for k in range(N):
                if k == 0:
                    ego_vehicle_pose = waypoints[0,0:2]
                    yaw_ego = waypoints[0, 2]
                    continue

                curr_yaw = waypoints[k, 2]
                curr_position = waypoints[k, 0:2]

                # make waypoints wrt ego vehicle 
                rel_pose = curr_position - ego_vehicle_pose
                rel_pose = rel_pose.reshape((2, 1))
                rel_pose_pix = np.copy(rel_pose)
                rel_pose_pix[0] = int(rel_pose_pix[0] / resolution)
                rel_pose_pix[1] = int(rel_pose_pix[1] / resolution)
                
                # Determine rotation needed for waypoint
                theta = np.pi - yaw_ego + (3*np.pi/180)

                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                # XY/px positions
                rot_pose_pix = np.matmul(R, rel_pose_pix)
                rot_pose_xy = np.matmul(R, rel_pose)

                rot_pose_x = rot_pose_xy[0]
                rot_pose_y = rot_pose_xy[1]
                rot_pose_u = rot_pose_pix[0]
                rot_pose_v = rot_pose_pix[1]

                # Waypoints XY
                waypoints_x.append(rot_pose_x)
                waypoints_y.append(rot_pose_y)

                # Waypoints px
                waypoints_px_u.append(rot_pose_u)
                waypoints_px_v.append(rot_pose_v)


            waypoints_x = np.asarray(waypoints_x)
            waypoints_y = np.asarray(waypoints_y)

            waypoints_px_u = np.asarray(waypoints_px_u)
            waypoints_px_v = np.asarray(waypoints_px_v)
            
            waypoints_xy = np.hstack((waypoints_x, waypoints_y)).T
            waypoints_px = np.hstack((waypoints_px_u, waypoints_px_v)).T

            # Load semantic bev 
            lbpsm_map = cv2.imread(data_segment + "/semantic/" + str(j) + ".png")
            lbpsm_cp = np.copy(lbpsm_map)

            # Load routed osm map 
            osm_map = cv2.imread(data_segment + "/routed/" + str(j) + ".png")

            if semantic_tf and osm_tf:
                lbpsm_map = semantic_tf(lbpsm_map)
                osm_map = osm_tf(osm_map)
                if use_gpu and torch.cuda.is_available():
                    lbpsm_map = lbpsm_map.to("cuda")
                    osm_map = osm_map.to("cuda")

            #loaded_data.append([j, segment_type, lbpsm_map, osm_map, waypoints_px, waypoints_xy])
            loaded_data.append([j, data_segment, lbpsm_map, osm_map, waypoints_px, waypoints_xy])

            # Save visualization
            if save_visualization:
                
                for k in range(waypoints_px.shape[1]):
                    lbpsm_cp = cv2.circle(lbpsm_cp, 
                            (int(waypoints_px[1, k]+cfg["nn_params"]["ego_center_v"]), 
                             int(waypoints_px[0, k]+cfg["nn_params"]["ego_center_u"])),
                            1, (255,0,0), -1)
                cv2.imwrite("lbpsm_viz/" + str(j) + ".png", lbpsm_cp)
            # Assert the number of waypoints generate equals horizon
    
    print("Data summary:")
    print("Number of samples: " + str(len(loaded_data)))
    return loaded_data

def get_ego_trajectories(data_path, cfg, save_visualization=False, semantic_tf=None, osm_tf=None, use_gpu=True):


    #x_origin = cfg["maps"]["x_origin"]
    #y_origin = cfg["maps"]["y_origin"]
    resolution = cfg["maps"]["resolution"]
    ego_speed_thresh = cfg["thresholds"]["ego_vehicle_speed"]
    waypoint_speed_thresh = cfg["thresholds"]["waypoint_speed"]
    waypoint_dist_thresh = cfg["thresholds"]["waypoint_dist_threshold"]
    horizon = cfg["nn_params"]["horizon"]
    
    #loaded_data = [[] for i in range(data_labels_np.shape[0])]
    loaded_data = []
    # 0: NS; 1:IS; 2:IL; 3:IR
    if save_visualization:
        os.mkdir("lbpsm_viz")


    # Read lbpsm data between range_start and range_end and extract feasible trajectories
    for data_segment in glob.iglob(data_path + '/**'):
        poses = np.load(data_segment + "/poses.npy")
        speed = np.load(data_segment + "/speed.npy")

        for j in range(poses.shape[0]):
            waypoints = poses[j:]
            current_speed = speed[j:]
            in_between_distance = 0
            prev_waypoint = 0
            new_waypoints = waypoints[0,:].reshape((1,7))
            N = waypoints.shape[0]
            
            terminate_sequence = False
            # If ego-vehicle is moving relatively slow or there are not enough waypoints, continue
            if current_speed[0] <= ego_speed_thresh or N < horizon:
                continue
            # Collect horizon number of waypoints, unless there is a discontinuity caused 
            # by waypoints with low speed
            #for k in range(1, min(3*horizon, N), 1):
            for k in range(1, N, 1):
                #distance = waypoints[prev_waypoint, 0:2] - waypoints[k, 0:2]
                distance = waypoints[k, 0:2] - waypoints[prev_waypoint, 0:2]
                curr_speed = current_speed[k]

                # Disregards current sequence if speed at waypoint k too low
                if curr_speed < waypoint_speed_thresh:
                    terminate_sequence = True
                    j = k
                    break

                in_between_distance += np.sqrt(np.sum(distance ** 2))
                if (in_between_distance >= waypoint_dist_thresh) and (curr_speed >= waypoint_speed_thresh):
                    new_waypoints = np.vstack((new_waypoints, waypoints[k,:].reshape((1,7))))
                    prev_waypoint = k
                    in_between_distance = 0

                if new_waypoints.shape[0] == horizon+1:
                    break
            
            # Executed if sequence does not have enough waypoints
            if terminate_sequence:
                termiante_sequence = False
                continue

            waypoints = new_waypoints
            N = waypoints.shape[0]
            waypoints_px_u = []
            waypoints_px_v = []

            if N != horizon+1:
                terminate_sequence = False
                continue

            waypoints_x = []
            waypoints_y = []
            ego_vehicle_pose = []
            yaw_ego = 0
            
            # Transform waypoints into ego-vehicle frame and pixel coordinates
            for k in range(N):
                if k == 0:
                    ego_vehicle_pose = waypoints[0,0:2]
                    yaw_ego = get_yaw(waypoints[0, 3:], False)
                    continue

                curr_yaw = get_yaw(waypoints[k, 3:], False)
                curr_position = waypoints[k, 0:2]

                # make waypoints wrt ego vehicle 
                rel_pose = curr_position - ego_vehicle_pose
                rel_pose = rel_pose.reshape((2, 1))
                rel_pose_pix = np.copy(rel_pose)
                rel_pose_pix[0] = int(rel_pose_pix[0] / resolution)
                rel_pose_pix[1] = int(rel_pose_pix[1] / resolution)
                
                # Determine rotation needed for waypoint
                theta = np.pi - yaw_ego + (3*np.pi/180)

                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                # XY/px positions
                rot_pose_pix = np.matmul(R, rel_pose_pix)
                rot_pose_xy = np.matmul(R, rel_pose)

                rot_pose_x = rot_pose_xy[0]
                rot_pose_y = rot_pose_xy[1]
                rot_pose_u = rot_pose_pix[0]
                rot_pose_v = rot_pose_pix[1]

                # Waypoints XY
                waypoints_x.append(rot_pose_x)
                waypoints_y.append(rot_pose_y)

                # Waypoints px
                waypoints_px_u.append(rot_pose_u)
                waypoints_px_v.append(rot_pose_v)


            waypoints_x = np.asarray(waypoints_x)
            waypoints_y = np.asarray(waypoints_y)

            waypoints_px_u = np.asarray(waypoints_px_u)
            waypoints_px_v = np.asarray(waypoints_px_v)
            
            waypoints_xy = np.hstack((waypoints_x, waypoints_y)).T
            waypoints_px = np.hstack((waypoints_px_u, waypoints_px_v)).T

            # Load semantic bev 
            lbpsm_map = cv2.imread(data_segment + "/semantic/" + str(j) + ".png")
            lbpsm_cp = np.copy(lbpsm_map)

            # Load routed osm map 
            osm_map = cv2.imread(data_segment + "/routed/" + str(j) + ".png")

            if semantic_tf and osm_tf:
                lbpsm_map = semantic_tf(lbpsm_map)
                osm_map = osm_tf(osm_map)
                if use_gpu and torch.cuda.is_available():
                    lbpsm_map = lbpsm_map.to("cuda")
                    osm_map = osm_map.to("cuda")

            #loaded_data.append([j, segment_type, lbpsm_map, osm_map, waypoints_px, waypoints_xy])
            loaded_data.append([j, data_segment, lbpsm_map, osm_map, waypoints_px, waypoints_xy])

            # Save visualization
            if save_visualization:
                
                for k in range(waypoints_px.shape[1]):
                    lbpsm_cp = cv2.circle(lbpsm_cp, 
                            (int(waypoints_px[1, k]+cfg["nn_params"]["ego_center_v"]), 
                             int(waypoints_px[0, k]+cfg["nn_params"]["ego_center_u"])),
                            1, (255,0,0), -1)
                cv2.imwrite("lbpsm_viz/" + str(j) + ".png", lbpsm_cp)
            # Assert the number of waypoints generate equals horizon
    
    print("Data summary:")
    print("Number of samples: " + str(len(loaded_data)))
    return loaded_data


def lsq_control_points(pts, num_c_points):
    n = pts.shape[1]
    P = np.zeros((n, num_c_points))
    X = np.arange(n) / (n-1)

    choose = lambda a, b : math.factorial(a) / (math.factorial(a - b) * math.factorial(b))

    
    #for i in range(n):
    #    for j in range(num_c_points):
    #        P[i,j] = choose(num_c_points - 1, j)*np.power(1-t[i],num_c_points - 1 - j)*np.power(t[i],j)

    weights = []
    for x in X:
        weight_x = [choose(num_c_points - 1, k) * ( np.power(1-x, num_c_points - 1 - k) * (x**k) ) for k in range(0, num_c_points)]
        weights.append(weight_x)

    weights = np.asarray(weights)
    
    return np.linalg.lstsq(weights, pts.T, rcond=None)[0]


## Ignore this function for now
def parameterize_curve(waypoints, num_control_points):
    """
    waypoints: (2, N)
    num_control_points: 2 < num_control_points < N/2
    """
    N = waypoints.shape[1]
    step = int(np.ceil(waypoints.shape[1] / num_control_points))
    control_pts = []

    for i in range(0, N, step):
        control_pts.append(waypoints[:, i])
    return np.asarray(control_pts)


def generate_bezier(control_points, n):
    """
    control_points: Control points that characterize Bezier curve
        np.array: (2, num_control_pts)
    n: number of points to generate
    """

    step = 1.0 / n 

    # estimate weight matrix (T, num_control_pts) (num_control_pts, 2)
    # (2, num_control_pts) (num_control_points, T)
    num_c_points = control_points.shape[1]
    choose = lambda a, b : math.factorial(a) / (math.factorial(a - b) * math.factorial(b))
    weights = []

    for x in np.arange(0, 1, step):
        weight_x = [choose(num_c_points - 1, k) * ( np.power(1-x, num_c_points - 1 - k) * (x**k) ) for k in range(0, num_c_points)]
        weights.append(weight_x)
    
    # (num_control_points, T)
    weights = np.asarray(weights).T


    return control_points @ weights


def get_osm_bezier(data_path, cfg, save_visualization=False, semantic_tf=None, use_gpu=True):


    resolution = cfg["maps"]["resolution"]
    waypoint_horizon = cfg["nn_params"]["horizon"]
    inbetween_thresh = cfg["interpolation_thresholds"]["inbetween_distance"]
    density = cfg["interpolation_thresholds"]["density"]
    speed_thresh = cfg["interpolation_thresholds"]["speed_threshold"]
    distance_horizon = (waypoint_horizon * inbetween_thresh) 
    
    loaded_data = []

    if save_visualization:
        os.mkdir("lbpsm_viz")
        os.mkdir("osm_graph_viz")


    curr_json = None

    # Read lbpsm data between range_start and range_end and extract feasible trajectories
    for data_segment in glob.iglob(data_path + '/**'):

        #json_files = data_path + "graphs/"
        json_files = data_segment + "/graphs/"
        num_samples = len([name for name in os.listdir(json_files) if os.path.isfile(os.path.join(json_files, name))])


        # process json file
        poses = np.zeros((num_samples, 7)) 
        speed = np.zeros(num_samples)

        for j in range(num_samples):
            with open(json_files + str(j) + '.json') as f:
                curr_json = json.load(f)
                
            poses[j, :] = np.asarray(curr_json['pose']) 
            speed[j] = curr_json['speed']
        
             
            
        for j in range(poses.shape[0]):
            with open(json_files + str(j) + '.json') as f:
                curr_json = json.load(f)
                

            # retrieve traversed/planned trajectories from OSM representation
            traversed_traj, planned_traj = get_osm_trajectories(curr_json, 20, 20)
            if save_visualization:
                visualize_graph(j, traversed_traj, planned_traj)

            # begin interpolation process
            waypoints = poses[j:]
            current_speed = speed[j:]
            current_distance = 0
            prev_waypoint = 0

            ego_vehicle_pose = waypoints[0,0:2]
            yaw_ego = get_yaw(waypoints[0, 3:], False)

            interpolated_wp = np.hstack((ego_vehicle_pose, yaw_ego)).reshape((1,3))
            N = waypoints.shape[0]
            
            # If ego-vehicle is moving relatively slow, continue
            if current_speed[0] <= speed_thresh:
                continue

            # Collect enough waypoints to cover 'distance_horizon' meters and
            # interpolate
            for k in range(1, N, 1):
                distance = waypoints[k, 0:2] - waypoints[prev_waypoint, 0:2]
                curr_speed = current_speed[k]
                dx = np.sqrt(np.sum(distance ** 2))
                
                if dx <= density:
                    continue

                current_distance += dx

                # Interpolate
                x0 = waypoints[prev_waypoint, 0]
                y0 = waypoints[prev_waypoint, 1]
                yaw0 = get_yaw(waypoints[prev_waypoint, 3:], False)

                xf = waypoints[k, 0]
                yf = waypoints[k, 1]
                yawf = get_yaw(waypoints[k, 3:], False)

                N_s = int(dx/density)
                inter_nodes = []
                for s in range(N_s):
                    x_s = ((N_s - s)*x0 + (s + 1)*xf)/(N_s + 1)
                    y_s = ((N_s - s)*y0 + (s + 1)*yf)/(N_s + 1)
                    yaw_s = ((N_s - s)*yaw0 + (s + 1)*yawf)/(N_s + 1)
                    inter_nodes.append(np.hstack((x_s, y_s, yaw_s)).reshape((1,3)))

                inter_nodes.append(np.hstack((xf, yf, yawf)).reshape((1,3)))
                inter_nodes = np.vstack(inter_nodes)    
                interpolated_wp = np.vstack((interpolated_wp, inter_nodes))

                prev_waypoint = k

                if current_distance >= 2*distance_horizon:
                    break

            # Retrive final set of waypoints spaced 'inbetween_thresh' apart 
            curr_inbetween = 0
            prev_waypoint = 0
            final_wp = []
            final_wp.append(interpolated_wp[0, 0:3].reshape((1,3)))

            for k in range(1, interpolated_wp.shape[0], 1):
                distance = interpolated_wp[k, 0:2] - interpolated_wp[prev_waypoint, 0:2]
                dx = np.sqrt(np.sum(distance ** 2))
                #print("dx: " + str(dx))
                
                #curr_inbetween += dx
                if dx >= inbetween_thresh:
                    #print("curr_inbetween: " + str(curr_inbetween))
                    final_wp.append(interpolated_wp[k, 0:3].reshape((1,3)))
                    prev_waypoint = k
                    curr_inbetween = 0

                if len(final_wp) == waypoint_horizon+1:
                    break
                
            final_wp = np.vstack(final_wp)
                     
            waypoints = final_wp 
            N = waypoints.shape[0]
            waypoints_px_u = []
            waypoints_px_v = []

            waypoints_x = []
            waypoints_y = []
            ego_vehicle_pose = []
            yaw_ego = 0

            if N != waypoint_horizon+1:
                continue
            
            # Transform waypoints into ego-vehicle frame and pixel coordinates
            for k in range(N):
                if k == 0:
                    ego_vehicle_pose = waypoints[0,0:2]
                    yaw_ego = waypoints[0, 2]
                    continue

                curr_yaw = waypoints[k, 2]
                curr_position = waypoints[k, 0:2]

                # make waypoints wrt ego vehicle 
                rel_pose = curr_position - ego_vehicle_pose
                rel_pose = rel_pose.reshape((2, 1))
                
                #rel_pose_pix = np.copy(rel_pose)
                #rel_pose_pix[0] = int(rel_pose_pix[0] / resolution)
                #rel_pose_pix[1] = int(rel_pose_pix[1] / resolution)
                
                # Determine rotation needed for waypoint
                theta = np.pi - yaw_ego + (3*np.pi/180)

                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                # XY/px positions

                rot_pose_xy = np.matmul(R, rel_pose)

                rot_pose_x = rot_pose_xy[0]
                rot_pose_y = rot_pose_xy[1]


                # Waypoints XY
                waypoints_x.append(rot_pose_x)
                waypoints_y.append(rot_pose_y)


            # waypoints wrt to ego
            waypoints_x = np.asarray(waypoints_x)
            waypoints_y = np.asarray(waypoints_y)

            
            waypoints_xy = np.hstack((waypoints_x, waypoints_y)).T
            
            # waypoints in BEV
            waypoints_px = waypoints_xy / resolution
            waypoints_px = waypoints_px.astype(int)

            # TODO: estimate bezier curve parameterization
            #control_points_xy = parameterize_curve(waypoints_xy, 5).T
            control_points_xy = lsq_control_points(waypoints_xy, 6).T
            b_curve = generate_bezier(control_points_xy, 30)
            #b_curve_px = b
            control_points_px = control_points_xy / resolution
            control_points_px = control_points_px.astype(int)
            b_curve_px = b_curve / resolution
            b_curve_px = b_curve_px.astype(int)


            # Load semantic bev 
            lbpsm_map = cv2.imread(data_segment + "/semantic/" + str(j) + ".png")
            lbpsm_cp = np.copy(lbpsm_map)


            if semantic_tf:
                lbpsm_map = semantic_tf(lbpsm_map)
                if use_gpu and torch.cuda.is_available():
                    lbpsm_map = lbpsm_map.to("cuda")

            waypoints_px_tf = torch.from_numpy(waypoints_px)
            waypoints_xy_tf = torch.from_numpy(waypoints_xy)
            traversed_tf = torch.from_numpy(traversed_traj.copy())
            planned_tf = torch.from_numpy(planned_traj.copy())

            if use_gpu and torch.cuda.is_available():
                waypoints_px_tf = waypoints_px_tf.float().cuda()                
                waypoints_xy_tf = waypoints_xy_tf.float().cuda()                
                traversed_tf = traversed_tf.float().cuda()                
                planned_tf = planned_tf.float().cuda()                

            loaded_data.append([j, data_segment, lbpsm_map, waypoints_px_tf, waypoints_xy_tf, 
                                traversed_tf, planned_tf])
            #loaded_data.append([j, data_segment, lbpsm_map, waypoints_px, waypoints_xy])

            # Save visualization (generated bezier)
            if save_visualization:
                
                vis_points = b_curve_px
                for k in range(vis_points.shape[1]):
                    lbpsm_cp = cv2.circle(lbpsm_cp, 
                            (int(vis_points[1, k]+cfg["nn_params"]["ego_center_v"]), 
                             int(vis_points[0, k]+cfg["nn_params"]["ego_center_u"])),
                            1, (255,0,0), -1)
                cv2.imwrite("lbpsm_viz/" + str(j) + ".png", lbpsm_cp)

            # Save visualization (interpolated waypoints)            
            if save_visualization:
                
                vis_points = waypoints_px
                for k in range(vis_points.shape[1]):
                    lbpsm_cp = cv2.circle(lbpsm_cp, 
                            (int(vis_points[1, k]+cfg["nn_params"]["ego_center_v"]), 
                             int(vis_points[0, k]+cfg["nn_params"]["ego_center_u"])),
                            1, (255,255,255), -1)
                cv2.imwrite("lbpsm_viz/" + str(j) + ".png", lbpsm_cp)
                # Save visualization (control points)
            if save_visualization:
                
                vis_points = control_points_px
                for k in range(vis_points.shape[1]):
                    lbpsm_cp = cv2.circle(lbpsm_cp, 
                            (int(vis_points[1, k]+cfg["nn_params"]["ego_center_v"]), 
                             int(vis_points[0, k]+cfg["nn_params"]["ego_center_u"])),
                            1, (0,255,255), -1)
                cv2.imwrite("lbpsm_viz/" + str(j) + ".png", lbpsm_cp)
            # Assert the number of waypoints generate equals horizon
    
    print("Data summary:")
    print("Number of samples: " + str(len(loaded_data)))
    return loaded_data

if __name__ == "__main__":
    
    data_path = "/mnt/00B0A680755C4DFA/DevSpace/DSM/Bezier/curve_parameterizations/bezier/data/03_sample/" 
    conf_path = "/mnt/00B0A680755C4DFA/DevSpace/DSM/Bezier/curve_parameterizations/bezier/config.yaml"

    with open(conf_path, "r") as conf_file:
        cfg = yaml.safe_load(conf_file)

    semantic_tf = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize((200, 200)),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x-0.5),
     transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )
     ])

    segments = get_osm_bezier(data_path, cfg, True, semantic_tf, False)