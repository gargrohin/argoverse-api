''' Code to generate json file for openPifPaf training
in a COCO dataset format.
Project: Dynamic Scene Modelling @ Autonomous Vehicles Lab UCSD
Author: Rohin Garg 
Github: @gargrohin'''


#################################

from argoverse.map_representation.map_api import ArgoverseMap
import json

import matplotlib
import matplotlib.pyplot as plt
from demo_usage.visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

import argoverse.visualization.visualization_utils as viz_util
from argoverse.utils.frustum_clipping import generate_frustum_planes
from PIL import Image 
from demo_usage.cuboids_to_bboxes import plot_lane_centerlines_in_img
import time
import logging



## JSON file template
def createJsonTemplate():
    data = {
        "info": {
        "description": "Argoverse test dataset",
        "url": "N/AS",
        "version": "0.0",
        "year": 2022,
        "contributor": "AVL",
        "date_created": "2022/01"
    },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ],
    "images": [],

    "annotations": [],

    "categories": [
        {
            "supercategory": "lane",
            "id": 1,
            "name": "lane control points",
            "keypoints": [
                "head", "2", "3","4","5","6","7","8","9", "tail"
            ],
            "skeleton": [
                [1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10]
            ]
        }
    ]

    }

    return data

def createDataset(am, data, argoverse_data):

    images_dicts = []
    lane_ann = []
    start = time.time()
    ann_id = -1
    cameras = ["ring_front_center", "ring_front_right", "ring_front_left",\
                "ring_rear_right", "ring_rear_left", "ring_side_right", "ring_side_left"]
    num_ctrl_pts = 10
    camera_id = -1            
    for camera in cameras:
        camera_id+=1
        calib = argoverse_data.get_calibration(camera = camera)
        planes = generate_frustum_planes(calib.camera_config.intrinsic.copy(), camera)
        for idx in range(156):
            image_id = idx + camera_id*100
            
            city_to_egovehicle_se3 = argoverse_data.get_pose(idx)

            name = argoverse_data.get_image_sync(idx,camera = camera, load=False)
            img = argoverse_data.get_image_sync(idx,camera = camera)
            img_dic = {
                "license": 1,
                "file_name": name,
                "coco_url": "NA",
                "height": img.shape[0],
                "width": img.shape[1],
                "date_captured": "idk",
                "id": image_id
            }
            images_dicts.append(img_dic)

            # objects = argoverse_data.get_label_object(idx)

            lidar_pts = argoverse_data.get_lidar(idx)

            img_wlane, lanes, lanes_bird = plot_lane_centerlines_in_img(lidar_pts, city_to_egovehicle_se3, img, city_name, am, calib.camera_config, planes)
            
            
            for lane in lanes:
                ann_id+=1
                if lane==[]:
                    continue
                dic = {
                    "num_keypoints": len(lane),
                    "area": 0,
                    "iscrowd": 0,
                    "keypoints": [],
                    "image_id": image_id,
                    "category_id": 1,
                    "id": ann_id
                }
                keypoints = []
                
                p = 0
                for c in range(num_ctrl_pts):
                    l = lane[p]
                    if l[1] != c:
                        for i in range(3):
                            keypoints.append(0)
                    else:
                        keypoints.append(l[0][0])
                        keypoints.append(l[0][1])
                        keypoints.append(2)
                        if p<len(lane)-1:
                            p+=1
                
                
                dic["keypoints"] = keypoints
                
                lane_ann.append(dic)
        
        print("time taken for camera ", camera, ": ", time.time()-start)
        # if visualize:
        #     display(Image.fromarray(img_wlane))
        start = time.time()

    data["images"] = images_dicts
    data["annotations"] = lane_ann

    return data

def dump_json(data, file):
    with open(file,"w") as write_file:
        json.dump(data, write_file)

if __name__ == "__main__":

    am = ArgoverseMap()

    tracking_dataset_dir = '../../argoverse_tracking_data/argoverse-tracking/train1/'

    # Map from a bird's-eye-view (BEV)
    # dataset_dir = tracking_dataset_dir
    # experiment_prefix = 'visualization_demo'
    # use_existing_files = True

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    log_index = 0
    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    log_id = argoverse_loader.log_list[log_index]
    argoverse_data = argoverse_loader[log_index]
    city_name = argoverse_data.city_name

    data = createJsonTemplate()
    data = createDataset(am, data, argoverse_data)
    json_path = "../../coco_format_train1_kp10.json"
    dump_json(data, json_path)

    print("DONE, file dumped to path: ", json_path)




    # domv = DatasetOnMapVisualizer(dataset_dir, experiment_prefix, use_existing_files=use_existing_files, log_id=argoverse_data.current_log)




