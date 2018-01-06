import numpy as np

import time
from itertools import chain
from collections import defaultdict

from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing



class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')
            
        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)



class RGBDPTAM(object):
    def __init__(self, params):
        self.params = params

        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params)
        self.tracker = Tracking(params)
        self.motion_model = MotionModel(params)

        self.loop_closing = LoopClosing(self, params)
        self.loop_correction = None
        
        self.reference = None        # reference keyframe
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.candidates = []         # candidate keyframes
        self.results = []            # tracking results

        self.status = defaultdict(bool)
        
    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def initialize(self, frame):
        mappoints, measurements = frame.cloudify()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

    def track(self, frame):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(
                    frame.orientation,
                    frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        print('measurements:', len(measurements), '   ', len(local_mappoints))

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)
        
        try:
            self.reference = self.graph.get_reference_frame(tracked_map)
            pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)

            frame.update_pose(pose)
            self.motion_model.update_pose(
                frame.timestamp, pose.position(), pose.orientation())
            # if len() > 40:
            self.candidates.append(frame)
            self.params.relax_tracking(False)
            self.results.append(True)    # tracking succeed
        except:
            self.params.relax_tracking(True)
            self.results.append(False)   # tracking fail

        remedy = False
        if self.results[-2:].count(False) == 2:
            if (len(self.candidates) > 0 and
                self.candidates[-1].idx > self.preceding.idx):
                frame = self.candidates[-1]
                remedy = True
            else:
                print('tracking failed!')

        if remedy or (
            self.results[-1] and self.should_be_keyframe(frame, measurements)):
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

            self.mapping.add_keyframe(keyframe, measurements)
            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)
            self.preceding = keyframe
            print('new keyframe', keyframe.idx)

        self.set_tracking(False)


    def filter_points(self, frame):
        # local_mappoints = self.graph.get_local_map(self.tracked_map)[0]
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        print('filter points:', len(local_mappoints), can_view.sum(), 
            len(self.preceding.mappoints()),
            len(self.reference.mappoints()))
        
        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered


    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False
        n_matches = len(measurements)
        return (n_matches < self.params.min_tracked_points and
            n_matches > self.params.pnp_min_measurements)


    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']





if __name__ == '__main__':
    import cv2
    import g2o

    import os
    import sys
    import argparse

    from threading import Thread
    
    from components import Camera
    from components import RGBDFrame
    from feature import ImageFeature
    from params import Params
    from dataset import TUMRGBDDataset, ICLNUIMDataset
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--dataset', type=str, default='ICL',
        help='dataset (TUM/ICL-NUIM)')
    # parser.add_argument('--path', type=str, help='dataset path', 
    #     default='path/to/your/TUM_RGBD/rgbd_dataset_freiburg1_room')
    parser.add_argument('--path', type=str, help='dataset path', 
        default='path/to/your/ICL-NUIM_RGBD/living_room_traj3_frei_png')
    args = parser.parse_args()



    if 'tum' in args.dataset.lower():
        dataset = TUMRGBDDataset(args.path)
    elif 'icl' in args.dataset.lower():
        dataset = ICLNUIMDataset(args.path)

    params = Params()
    ptam = RGBDPTAM(params)

    if not args.no_viz:
        from viewer import MapViewer
        viewer = MapViewer(ptam, params)

    height, width = dataset.rgb.shape[:2]
    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        width, height, dataset.cam.scale, 
        params.virtual_baseline, params.depth_near, params.depth_far,
        params.frustum_near, params.frustum_far)


    durations = []
    for i in range(len(dataset))[:]:
        feature = ImageFeature(dataset.rgb[i], params)
        depth = dataset.depth[i]
        if dataset.timestamps is None:
            timestamp = i / 20.
        else:
            timestamp = dataset.timestamps[i]

        time_start = time.time()  
        feature.extract()

        frame = RGBDFrame(i, g2o.Isometry3d(), feature, depth, cam, timestamp=timestamp)

        if not ptam.is_initialized():
            ptam.initialize(frame)
        else:
            ptam.track(frame)


        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()
        print()
        
        if not args.no_viz:
            viewer.update()

    print('num frames', len(durations))
    print('num keyframes', len(ptam.graph.keyframes()))
    print('average time', np.mean(durations))


    ptam.stop()
    if not args.no_viz:
        viewer.stop()