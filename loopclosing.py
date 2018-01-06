import numpy as np
import cv2

import g2o
from g2o.contrib import SmoothEstimatePropagator

import time
from threading import Thread, Lock
from queue import Queue

from collections import defaultdict, namedtuple

from optimization import PoseGraphOptimization
from components import Measurement



# a very simple implementation
class LoopDetection(object):
    def __init__(self, params):
        self.params = params
        self.nns = NearestNeighbors()

    def add_keyframe(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        self.nns.add_item(embedding, keyframe)

    def detect(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        kfs, ds = self.nns.search(embedding, k=20)

        if len(kfs) > 0 and kfs[0] == keyframe:
            kfs, ds = kfs[1:], ds[1:]
        if len(kfs) == 0:
            return None

        min_d = np.min(ds)
        for kf, d in zip(kfs, ds):
            if abs(kf.id - keyframe.id) < self.params.lc_min_inbetween_keyframes:
                continue
            if (np.linalg.norm(kf.position - keyframe.position) > 
                self.params.lc_max_inbetween_distance):
                break
            if d > self.params.lc_embedding_distance or d > min_d * 1.5:
                break
            return kf
        return None



class LoopClosing(object):
    def __init__(self, system, params):
        self.system = system
        self.params = params

        self.loop_detector = LoopDetection(params)
        self.optimizer = PoseGraphOptimization()

        self.loops = []
        self.stopped = False

        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def stop(self):
        self.stopped = True
        self._queue.put(None)
        self.maintenance_thread.join()
        print('loop closing stopped')

    def add_keyframe(self, keyframe):
        self._queue.put(keyframe)
        self.loop_detector.add_keyframe(keyframe)

    def add_keyframes(self, keyframes):
        for kf in keyframes:
            self.add_keyframe(kf)

    def maintenance(self):
        last_query_keyframe = None
        while not self.stopped:
            keyframe = self._queue.get()
            if keyframe is None or self.stopped:
                return

            # check if this keyframe share many mappoints with a loop keyframe
            covisible = sorted(
                keyframe.covisibility_keyframes().items(), 
                key=lambda _:_[1], reverse=True)
            if any([(keyframe.id - _[0].id) > 5 for _ in covisible[:2]]):
                continue

            if (last_query_keyframe is not None and 
                abs(last_query_keyframe.id - keyframe.id) < 3):
                continue

            candidate = self.loop_detector.detect(keyframe)
            if candidate is None:
                continue

            query_keyframe = keyframe
            match_keyframe = candidate

            result = match_and_estimate(
                query_keyframe, match_keyframe, self.params)

            if result is None:
                continue
            if (result.n_inliers < max(self.params.lc_inliers_threshold, 
                result.n_matches * self.params.lc_inliers_ratio)):
                continue

            if (np.abs(result.correction.translation()).max() > 
                self.params.lc_distance_threshold):
                continue

            self.loops.append(
                (match_keyframe, query_keyframe, result.constraint))
            query_keyframe.set_loop(match_keyframe, result.constraint)

            # We have to ensure that the mapping thread is on a safe part of code, 
            # before the selection of KFs to optimize
            safe_window = self.system.mapping.lock_window()   # set
            safe_window.add(self.system.reference)
            for kf in self.system.reference.covisibility_keyframes():
                safe_window.add(kf)

            
            # The safe window established between the Local Mapping must be 
            # inside the considered KFs.
            considered_keyframes = self.system.graph.keyframes()

            self.optimizer.set_data(considered_keyframes, self.loops)

            before_lc = [
                g2o.Isometry3d(kf.orientation, kf.position) for kf in safe_window]

            # Propagate initial estimate through 10% of total keyframes 
            # (or at least 20 keyframes)
            d = max(20, len(considered_keyframes) * 0.1)
            propagator = SmoothEstimatePropagator(self.optimizer, d)
            propagator.propagate(self.optimizer.vertex(match_keyframe.id))

            # self.optimizer.set_verbose(True)
            self.optimizer.optimize(20)
            
            # Exclude KFs that may being use by the local BA.
            self.optimizer.update_poses_and_points(
                considered_keyframes, exclude=safe_window)

            self.system.stop_adding_keyframes()

            # Wait until mapper flushes everything to the map
            self.system.mapping.wait_until_empty_queue()
            while self.system.mapping.is_processing():
                time.sleep(1e-4)

            # Calculating optimization introduced by local mapping while loop was been closed
            for i, kf in enumerate(safe_window):
                after_lc = g2o.Isometry3d(kf.orientation, kf.position)
                corr = before_lc[i].inverse() * after_lc

                vertex = self.optimizer.vertex(kf.id)
                vertex.set_estimate(vertex.estimate() * corr)

            self.system.pause()

            for keyframe in considered_keyframes[::-1]:
                if keyframe in safe_window:
                    reference = keyframe
                    break
            uncorrected = g2o.Isometry3d(
                reference.orientation, 
                reference.position)
            corrected = self.optimizer.vertex(reference.id).estimate()
            T = uncorrected.inverse() * corrected   # close to result.correction

            # We need to wait for the end of the current frame tracking and ensure that we
            # won't interfere with the tracker.
            while self.system.is_tracking():
                time.sleep(1e-4)
            self.system.set_loop_correction(T)

            # Updating keyframes and map points on the lba zone
            self.optimizer.update_poses_and_points(safe_window)

            # keyframes after loop closing
            keyframes = self.system.graph.keyframes()
            if len(keyframes) > len(considered_keyframes):
                self.optimizer.update_poses_and_points(
                    keyframes[len(considered_keyframes) - len(keyframes):], 
                    correction=T)

            for query_meas, match_meas in result.shared_measurements:
                new_query_meas = Measurement(
                    query_meas.type,
                    Measurement.Source.REFIND,
                    query_meas.get_keypoints(),
                    query_meas.get_descriptors())
                self.system.graph.add_measurement(
                    query_keyframe, match_meas.mappoint, new_query_meas)
                
                new_match_meas = Measurement(
                    match_meas.type,
                    Measurement.Source.REFIND,
                    match_meas.get_keypoints(),
                    match_meas.get_descriptors())
                self.system.graph.add_measurement(
                    match_keyframe, query_meas.mappoint, new_match_meas)

            self.system.mapping.free_window()
            self.system.resume_adding_keyframes()
            self.system.unpause()

            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    return
            last_query_keyframe = query_keyframe
        


def depth_to_3d(depth, coords, cam):
    coords = np.array(coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    return np.column_stack([xs, ys, zs])


def match_and_estimate(query_keyframe, match_keyframe, params):
    query = defaultdict(list)
    for kp, desp in zip(
        query_keyframe.feature.keypoints, query_keyframe.feature.descriptors):
        query['kps'].append(kp)
        query['desps'].append(desp)
        query['px'].append(kp.pt)

    match = defaultdict(list)
    for kp, desp in zip(
        match_keyframe.feature.keypoints, match_keyframe.feature.descriptors):
        match['kps'].append(kp)
        match['desps'].append(desp)
        match['px'].append(kp.pt)

    matches = query_keyframe.feature.direct_match(
        query['desps'], match['desps'],
        params.matching_distance, 
        params.matching_distance_ratio)

    query_pts = depth_to_3d(query_keyframe.depth, query['px'], query_keyframe.cam)
    match_pts = depth_to_3d(match_keyframe.depth, match['px'], match_keyframe.cam)

    if len(matches) < params.lc_inliers_threshold:
        return None

    near = query_keyframe.cam.depth_near
    far = query_keyframe.cam.depth_far
    for (i, j) in matches:
        if (near <= query_pts[i][2] <= far):
            query['pt12'].append(query_pts[i])
            query['px12'].append(query['kps'][i].pt)
            match['px12'].append(match['kps'][j].pt)
        if (near <= match_pts[j][2] <= far):
            query['px21'].append(query['kps'][i].pt)
            match['px21'].append(match['kps'][j].pt)
            match['pt21'].append(match_pts[j])

    if len(query['pt12']) < 6 or len(match['pt21']) < 6:
        return None

    T12, inliers12 = solve_pnp_ransac(
        query['pt12'], match['px12'], match_keyframe.cam.intrinsic)

    T21, inliers21 = solve_pnp_ransac(
        match['pt21'], query['px21'], query_keyframe.cam.intrinsic)

    if T12 is None or T21 is None:
        return None

    delta = T21 * T12
    if (g2o.AngleAxis(delta.rotation()).angle() > 0.06 or
        np.linalg.norm(delta.translation()) > 0.06):          # 3° or 0.06m
        return None

    ms = set()
    qd = dict()
    md = dict()
    for i in inliers12:
        pt1 = (int(query['px12'][i][0]), int(query['px12'][i][1]))
        pt2 = (int(match['px12'][i][0]), int(match['px12'][i][1]))
        ms.add((pt1, pt2))
    for i in inliers21:
        pt1 = (int(query['px21'][i][0]), int(query['px21'][i][1]))
        pt2 = (int(match['px21'][i][0]), int(match['px21'][i][1]))
        ms.add((pt1, pt2))
    for i, (pt1, pt2) in enumerate(ms):
        qd[pt1] = i
        md[pt2] = i

    qd2 = dict()
    md2 = dict()
    for m in query_keyframe.measurements():
        pt = m.get_keypoint(0).pt
        idx = qd.get((int(pt[0]), int(pt[1])), None)
        if idx is not None:
            qd2[idx] = m
    for m in match_keyframe.measurements():
        pt = m.get_keypoint(0).pt
        idx = md.get((int(pt[0]), int(pt[1])), None)
        if idx is not None:
            md2[idx] = m
    shared_measurements = [(qd2[i], md2[i]) for i in (qd2.keys() & md2.keys())]

    n_matches = (len(query['pt12']) + len(match['pt21'])) / 2.
    n_inliers = max(len(inliers12), len(inliers21))
    query_pose = g2o.Isometry3d(
        query_keyframe.orientation, query_keyframe.position)
    match_pose = g2o.Isometry3d(
        match_keyframe.orientation, match_keyframe.position)

    # TODO: combine T12 and T21
    constraint = T12
    estimated_pose = match_pose * constraint
    correction = query_pose.inverse() * estimated_pose

    return namedtuple('MatchEstimateResult',
        ['estimated_pose', 'constraint', 'correction', 'shared_measurements', 
        'n_matches', 'n_inliers'])(
        estimated_pose, constraint, correction, shared_measurements, 
        n_matches, n_inliers)


def solve_pnp_ransac(pts3d, pts, intrinsic_matrix):
    val, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d), np.array(pts), 
            intrinsic_matrix, None, None, None, 
            False, 50, 2.0, 0.99, None)
    if inliers is None or len(inliers) < 5:
        return None, None

    T = g2o.Isometry3d(cv2.Rodrigues(rvec)[0], tvec)
    return T, inliers.ravel()
    


class NearestNeighbors(object):
    def __init__(self, dim=None):
        self.n = 0
        self.dim = dim
        self.items = dict()
        self.data = []
        if dim is not None:
            self.data = np.zeros((1000, dim), dtype='float32')

    def add_item(self, vector, item):
        assert vector.ndim == 1
        if self.n >= len(self.data):
            if self.dim is None:
                self.dim = len(vector)
                self.data = np.zeros((1000, self.dim), dtype='float32')
            else:
                self.data.resize(
                    (2 * len(self.data), self.dim) , refcheck=False)
        self.items[self.n] = item
        self.data[self.n] = vector
        self.n += 1

    def search(self, query, k):
        if len(self.data) == 0:
            return [], []

        ds = np.linalg.norm(query[np.newaxis, :] - self.data[:self.n], axis=1)
        ns = np.argsort(ds)[:k]
        return [self.items[n] for n in ns], ds[ns]