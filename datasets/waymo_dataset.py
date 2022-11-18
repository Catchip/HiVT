# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
# from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData

from math import pi as Pi
# for waymo decode
import numpy as np
import tensorflow as tf

# from google.protobuf import text_format
# from waymo_open_dataset.metrics.ops import py_metrics_ops
# from waymo_open_dataset.metrics.python import config_util_py as config_util
# from waymo_open_dataset.protos import motion_metrics_pb2
# from waymo_tutorial import _parse


class WaymoDataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'training'
            #self._directory = 'validation'
        elif split == 'validation':
            self._directory = 'validation'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        roadgraph_features = {
            'roadgraph_samples/dir':
                tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
            'roadgraph_samples/id':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/type':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/valid':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/xyz':
                tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
        }

        # Features of other agents.
        state_features = {
            'state/id':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/type':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/is_sdc':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/tracks_to_predict':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/current/bbox_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/height':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/length':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/timestamp_micros':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/valid':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/vel_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/width':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/z':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/future/bbox_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/height':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/length':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/timestamp_micros':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/valid':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/vel_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/width':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/z':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/past/bbox_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/height':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/length':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/timestamp_micros':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/valid':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/vel_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/width':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/z':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        }

        traffic_light_features = {
            'traffic_light_state/current/state':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/valid':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/id':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/x':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/y':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/z':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/past/state':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/valid':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/x':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/y':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/z':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/id':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        }

        self.features_description = {}
        self.features_description.update(roadgraph_features)
        self.features_description.update(state_features)
        self.features_description.update(traffic_light_features)
        self.features_description['scenario/id'] = tf.io.FixedLenFeature([1], tf.string, default_value=None)
        self.features_description['state/objects_of_interest'] = tf.io.FixedLenFeature([128], tf.int64, default_value=None)

        self.root = root
        # self._raw_file_names = os.listdir(self.raw_dir)
        # self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_file_names = os.listdir(self.processed_dir)
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]

        super(WaymoDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        if self._split == "train":
            # return os.path.join(self.root, self._directory,)
            return os.path.join(self.root, self._directory, 'split_0' )
        elif self._split == "validation":
            return os.path.join(self.root, self._directory,)

    @property
    def processed_dir(self) -> str:
        if self._split == "train":
            return os.path.join(self.root, self._directory, 'split_0_processed')
            # return os.path.join(self.root, self._directory, 'processed')
        elif self._split == "validation":
            return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        # am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            self.process_waymo(self._split, raw_path, self._local_radius)
            # data = TemporalData(**kwargs)
            # torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        # return len(self._raw_file_names)
        l = len(self._processed_file_names)
        return l

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


    # split = val or train
    # raw_path = path of single data file
    def process_waymo(self,
                          split: str,
                          raw_path: str,
                          # am: ArgoverseMap,
                          radius: float) -> Dict:
        dataset = tf.data.TFRecordDataset(raw_path, compression_type='')
        for data in dataset.as_numpy_iterator():
            kwargs = self.process_piece(data)
            if "wrong_data" in kwargs:
                continue
            temporaData = TemporalData(**kwargs)
            torch.save(temporaData, os.path.join(self.processed_dir, kwargs['seq_id'] + '.pt'))
        return kwargs

    def process_piece(self, data):
        decoded_example = tf.io.parse_single_example(data, self.features_description)

        past_states = tf.stack([
            decoded_example['state/past/x'],
            decoded_example['state/past/y'],
        ], -1)

        cur_states = tf.stack([
            decoded_example['state/current/x'],
            decoded_example['state/current/y'],
        ], -1)

        x = tf.concat([past_states, cur_states], 1)
        # x : [128, 11 , 2]
        

        y = tf.stack([
            decoded_example['state/future/x'],
            decoded_example['state/future/y'],
        ], -1)
        # y : [128 , 11 ,2]

        future_states = tf.stack([
            decoded_example['state/future/x'],
            decoded_example['state/future/y'],
        ], -1)

        lane_feature = tf.stack([
            decoded_example['roadgraph_samples/xyz']
        ])

        positions = tf.concat([past_states, cur_states, future_states], 1)

        past_is_valid = decoded_example['state/past/valid'] > 0
        current_is_valid = decoded_example['state/current/valid'] > 0
        future_is_valid = decoded_example['state/future/valid'] > 0
        gt_future_is_valid = tf.concat(
            [past_is_valid, current_is_valid, future_is_valid], 1)

        # gt_future_is_valid : [128, 91]

        # If a sample was not seen at all in the past, we declare the sample as
        # invalid.
        sample_is_valid = tf.reduce_any(
            tf.concat([past_is_valid, current_is_valid], 1), 1)

        seq_id = str(decoded_example['scenario/id'].numpy()[0])
        actor_id = decoded_example['state/id']

        flag = True

        index_transformer= [dict(), dict()]

        num_nodes = 0
        for i in range(len(actor_id)):
            if actor_id[i] != -1:
                index_transformer[0][num_nodes] = i
                index_transformer[1][i] = num_nodes
                if num_nodes == 0:
                    X = [x[i]]
                    Y = [y[i]]
                    Positions = [positions[i]]
                else:
                    X.append(x[i])
                    Y.append(y[i])
                    Positions.append(positions[i])
                num_nodes = num_nodes + 1

        X = tf.stack(X, 0)
        Y = tf.stack(Y, 0)
        Positions = tf.stack(Positions, 0)

        x = torch.from_numpy(X.numpy())
        y = torch.from_numpy(Y.numpy())
        positions = torch.from_numpy(Positions.numpy())

        
        av_index = list(decoded_example['state/is_sdc']).index(1)
        av_index = index_transformer[1][av_index]


        agent_indices = np.where(decoded_example['state/tracks_to_predict'].numpy() > 0 )[0]
        agent_indices = [index_transformer[1][i] for i in agent_indices]




        vel_yaw = decoded_example['state/current/vel_yaw']
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        # padding_mask: [num_nodes, 91]
        # bos_mask: [num_nodes, 11]
        # rotate_angles: [num_nodes]
        padding_mask = torch.ones(num_nodes, 91, dtype=torch.bool)
        bos_mask = torch.zeros(num_nodes, 11, dtype=torch.bool)

        for i in range(num_nodes):
            index = index_transformer[0][i]
            padding_mask[i] = ~ torch.from_numpy(gt_future_is_valid[index].numpy())
            rotate_angles[i] = torch.from_numpy(vel_yaw[index].numpy())
            i = i + 1

        origin = torch.tensor([x[av_index, 10, 0], x[av_index, 10, 1]], dtype=torch.float)
        # print(rotate_angles)
        av_heading_theta = rotate_angles[av_index]
        rotate_mat = torch.tensor([[torch.cos(av_heading_theta), -torch.sin(av_heading_theta)],
                                   [torch.sin(av_heading_theta), torch.cos(av_heading_theta)]])

        #set the scene centered at av

        for node_index in range(num_nodes):
            duration = len(x[node_index])
            for timestamp in range(duration):
                xy = x[node_index, timestamp]
                x[node_index, timestamp] = torch.matmul(xy - origin, rotate_mat)
            rotate_angles[node_index] = rotate_angles[node_index] - av_heading_theta

            duration = len(y[node_index])
            for timestamp in range(duration):
                xy = y[node_index, timestamp]
                y[node_index, timestamp] = torch.matmul(xy - origin, rotate_mat)
            duration = len(positions[node_index])
            for timestamp in range(duration):
                xy = positions[node_index, timestamp]
                positions[node_index, timestamp] = torch.matmul(xy - origin, rotate_mat)
            if padding_mask[node_index, 10]:
                padding_mask[node_index, 11] = True

        bos_mask[:,0] = ~ padding_mask[:,0]
        bos_mask[:,1:11] = padding_mask[:,:10] & ~ padding_mask[:,1:11]

        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()

        y = torch.where((padding_mask[:, 10].unsqueeze(-1) | padding_mask[:, 11:]).unsqueeze(-1),
                                torch.zeros(num_nodes, 80, 2),
                                y - x[:, 10].unsqueeze(-2))

        x[:, 1: 11] = torch.where((padding_mask[:, : 10] | padding_mask[:, 1: 11]).unsqueeze(-1),
                                  torch.zeros(num_nodes, 10, 2),
                                  x[:, 1: 11] - x[:, : 10])
        x[:, 0] = torch.zeros(num_nodes, 2)


        # (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
         # lane_actor_vectors) = get_lane_features(node_inds_10, node_positions_10, origin, rotate_mat, city, radius)

        roadgraph_samples_xy = decoded_example['roadgraph_samples/xyz'][:,:2].numpy()
        roadgraph_samples_valid = decoded_example['roadgraph_samples/valid'].numpy()
        roadgraph_samples_id = decoded_example['roadgraph_samples/id'].numpy()
        roadgraph_samples_type = decoded_example['roadgraph_samples/type'].numpy()
        roadgraph_samples_dir = decoded_example['roadgraph_samples/dir'].numpy()
        road_ids = np.unique(roadgraph_samples_id)
        roadgraph_samples_num = len(roadgraph_samples_xy)
        lane_polylines = dict()
        lane_dir_pairs = dict()


        numbytype = [0] * 20
        for i in range(roadgraph_samples_num):
            # only centerline count
            sample_id = roadgraph_samples_id[i][0]
            if roadgraph_samples_valid[i] != 1 or sample_id == -1 or roadgraph_samples_type[i][0] > 3:
                continue
            if sample_id in lane_polylines:
                xy = roadgraph_samples_xy[i]
                lane_polylines[sample_id].append(roadgraph_samples_xy[i])
            else:
                xy = roadgraph_samples_xy[i]
                lane_polylines[sample_id] = [roadgraph_samples_xy[i]]

        if len(lane_polylines) == 0:
            return {"wrong_data": True}
        lane_vectors = []
        lane_positions = []
        is_intersections = []
        traffic_controls = []
        turn_directions = []

        for lane_id, lane_polyline in lane_polylines.items():
            lane_polyline = torch.from_numpy(np.array(lane_polyline))
            lane_polyline = torch.matmul(lane_polyline - origin, rotate_mat)
            lane_vectors.append(lane_polyline[1:] - lane_polyline[:-1])
            lane_positions.append(lane_polyline[:-1])
            count = len(lane_polyline) - 1

            if len(lane_polyline) < 2:
                turn_direction = 0
                is_intersection = False
                traffic_control = False
            else:
                dir1 = (lane_polyline[1] - lane_polyline[0]).numpy()
                dir2 = (lane_polyline[-1] - lane_polyline[-2]).numpy()
                theNorm = np.linalg.norm(dir1) * np.linalg.norm(dir2)
                rho = np.rad2deg(np.arcsin(np.cross(dir1, dir2)/ theNorm ))
                theta = np.rad2deg(np.arccos(np.dot(dir1, dir2)/ theNorm))

                theta = theta if rho > 0 else -theta

                if theta > 60 and theta < 180:
                    turn_direction = 1
                elif theta < -60 and theta > -180:
                    turn_direction = 2
                else:
                    turn_direction = 0

                is_intersection = True if abs(theta) > 60 else False

                traffic_control = is_intersection

            turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
            traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
            is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))


        lane_positions = torch.cat(lane_positions, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)
        is_intersections = torch.cat(is_intersections, dim=0)
        turn_directions = torch.cat(turn_directions, dim=0)
        traffic_controls = torch.cat(traffic_controls, dim=0)

        actor_current_inds = np.where(decoded_example['state/current/valid'].numpy() > 0)[0]
        actor_current_inds = [index_transformer[1][index] for index in actor_current_inds]
        actor_current_position = []
        for index in actor_current_inds:
            actor_current_position.append(x[index][10])
        actor_current_position = torch.stack(actor_current_position, dim=0)

        # print(lane_vectors.size(0))
        # print(actor_current_inds)
        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), actor_current_inds))).t().contiguous()

        lane_actor_vectors = \
            lane_positions.repeat_interleave(len(actor_current_inds), dim=0) - actor_current_position.repeat(lane_vectors.size(0), 1)

        radius = 150
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
        # print(seq_id)

        return {
            'x': x,  # [N, 20, 2]
            'positions': positions,  # [N, 50, 2]
            'edge_index': edge_index,  # [2, N x N - 1]
            'y': y,  # [N, 30, 2]
            'num_nodes': num_nodes,
            'padding_mask': padding_mask,  # [N, 50]
            'bos_mask': bos_mask,  # [N, 20]
            'rotate_angles': rotate_angles,  # [N]
            'lane_vectors': lane_vectors,  # [L, 2]
            'is_intersections': is_intersections,  # [L]
            'turn_directions': turn_directions,  # [L]
            'traffic_controls': traffic_controls,  # [L]
            'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
            'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
            'seq_id': seq_id,
            'av_index': av_index,
            'agent_index': agent_indices,
        }



