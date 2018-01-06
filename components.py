import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict

from covisibility import GraphKeyFrame
from covisibility import GraphMapPoint
from covisibility import GraphMeasurement




class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height, 
            scale, baseline, depth_near, depth_far, 
            frustum_near, frustum_far):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        self.baseline = baseline
        self.bf = fx * baseline

        self.intrinsic = np.array([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]])

        self.depth_near = depth_near
        self.depth_far = depth_far
        self.frustum_near = frustum_near
        self.frustum_far = frustum_far

        self.width = width
        self.height = height



class Frame(object):
    def __init__(self, idx, pose, feature, cam, timestamp=None, 
            pose_covariance=np.identity(6)):
        self.idx = idx
        self.pose = pose    # g2o.Isometry3d
        self.feature = feature
        self.cam = cam
        self.timestamp = timestamp
        self.image = feature.image
        
        self.orientation = pose.orientation()
        self.position = pose.position()
        self.pose_covariance = pose_covariance

        self.transform_matrix = pose.inverse().matrix()[:3] # shape: (3, 4)
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))  # from world frame to image

    def can_view(self, points, margin=10):    # Frustum Culling (batch version)
        points = np.transpose(points)
        (u, v), depth = self.project(self.transform(points))
        return np.logical_and.reduce([
            depth >= self.cam.frustum_near,
            depth <= self.cam.frustum_far,
            u >= - margin,
            u <= self.cam.width + margin,
            v >= - margin,
            v <= self.cam.height + margin])
        
    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose   
        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))

    def transform(self, points):    # from world coordinates
        '''
        Transform points from world coordinates frame to camera frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        assert len(points) > 0
        R = self.transform_matrix[:3, :3]
        if points.ndim == 1:
            t = self.transform_matrix[:3, 3]
        else:
            t = self.transform_matrix[:3, 3:]
        return R.dot(points) + t

    def project(self, points): 
        '''
        Project points from camera frame to image's pixel coordinates.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        Returns:
            Projected pixel coordinates, and respective depth.
        '''
        projection = self.cam.intrinsic.dot(points / points[-1:])
        return projection[:2], points[-1]

    def find_matches(self, points, descriptors):
        '''
        Match to points from world frame.
        Args:
            points: a list/array of points. shape: (N, 3)
            descriptors: a list of feature descriptors. length: N
        Returns:
            List of successfully matched (queryIdx, trainIdx) pairs.
        '''
        points = np.transpose(points)
        proj, _ = self.project(self.transform(points))
        proj = proj.transpose()
        return self.feature.find_matches(proj, descriptors)

    def get_keypoint(self, i):
        return self.feature.get_keypoint(i)
    def get_descriptor(self, i):
        return self.feature.get_descriptor(i)
    def get_color(self, pt):
        return self.feature.get_color(pt)
    def set_matched(self, i):
        self.feature.set_matched(i)
    def get_unmatched_keypoints(self):
        return self.feature.get_unmatched_keypoints()



def depth_to_3d(depth, coords, cam):
    coords = np.array(coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    return np.column_stack([xs, ys, zs])



class RGBDFrame(Frame):
    def __init__(self, idx, pose, feature, depth, cam, timestamp=None, 
            pose_covariance=np.identity(6)):

        super().__init__(idx, pose, feature, cam, timestamp, pose_covariance)
        self.rgb  = Frame(idx, pose, feature, cam, timestamp, pose_covariance)
        self.depth = depth

    def virtual_stereo(self, px):
        x, y = int(px[0]), int(px[1])
        if not (0 <= x <= self.cam.width-1 and 0 <= y <= self.cam.height-1):
            return None
        depth = self.depth[y, x] / self.cam.scale
        if not (self.cam.depth_near <= depth <= self.cam.depth_far):
            return None
        disparity = self.cam.bf / depth

        # virtual right camera observation
        kp2 = cv2.KeyPoint(x - disparity, y, 1) 
        return kp2

    def find_matches(self, source, points, descriptors):
        matches = self.rgb.find_matches(points, descriptors)
        measurements = []
        for i, j in matches:
            px = self.rgb.get_keypoint(j).pt
            kp2 = self.virtual_stereo(px)
            if kp2 is not None:
                measurement = Measurement(
                    Measurement.Type.STEREO,
                    source,
                    [self.rgb.get_keypoint(j), kp2],
                    [self.rgb.get_descriptor(j)] * 2)
            else:
                measurement = Measurement(
                    Measurement.Type.LEFT,
                    source,
                    [self.rgb.get_keypoint(j)],
                    [self.rgb.get_descriptor(j)])
            measurements.append((i, measurement))
            self.rgb.set_matched(j)

        return measurements

    def match_mappoints(self, mappoints, source):
        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)
        matched_measurements = self.find_matches(source, points, descriptors)

        measurements = []
        for i, meas in matched_measurements:
            meas.mappoint = mappoints[i]
            measurements.append(meas)
        return measurements

    def cloudify(self):
        kps, desps, idx = self.rgb.get_unmatched_keypoints()
        px = np.array([kp.pt for kp in kps])
        if len(px) == 0:
            return [], []

        pts = depth_to_3d(self.depth, px, self.cam)
        Rt = self.pose.matrix()[:3]
        R = Rt[:, :3]
        t = Rt[:, 3:]
        points = (R.dot(pts.T) + t).T   # world frame

        mappoints = []
        measurements = []
        for i, point in enumerate(points):
            if not (self.cam.depth_near <= pts[i][2] <= self.cam.depth_far):
                continue
            kp2 = self.virtual_stereo(px[i])
            if kp2 is None:
                continue

            normal = point - self.position
            normal /= np.linalg.norm(normal)
            color = self.rgb.get_color(px[i])

            mappoint = MapPoint(point, normal, desps[i], color)
            measurement = Measurement(
                Measurement.Type.STEREO,
                Measurement.Source.TRIANGULATION,
                [kps[i], kp2],
                [desps[i], desps[i]])
            measurement.mappoint = mappoint
            mappoints.append(mappoint)
            measurements.append(measurement)
            self.rgb.set_matched(i)
        return mappoints, measurements
            
    def update_pose(self, pose):
        super().update_pose(pose)
        self.rgb.update_pose(pose)

    def can_view(self, mappoints):  # batch version
        points = []
        point_normals = []
        for i, p in enumerate(mappoints):
            points.append(p.position)
            point_normals.append(p.normal)
        points = np.asarray(points)
        point_normals = np.asarray(point_normals)

        normals = points - self.position
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        cos = np.clip(np.sum(point_normals * normals, axis=1), -1, 1)

        parallel = np.arccos(cos) < (np.pi / 4)
        can_view = self.rgb.can_view(points)
        return np.logical_and(parallel, can_view)

    def to_keyframe(self):
        return KeyFrame(
            self.idx, self.pose, 
            self.feature, self.depth, 
            self.cam, self.pose_covariance)



class KeyFrame(GraphKeyFrame, RGBDFrame):
    _id = 0
    _id_lock = Lock()

    def __init__(self, *args, **kwargs):
        GraphKeyFrame.__init__(self)
        RGBDFrame.__init__(self, *args, **kwargs)

        with KeyFrame._id_lock:
            self.id = KeyFrame._id
            KeyFrame._id += 1

        self.reference_keyframe = None
        self.reference_constraint = None
        self.preceding_keyframe = None
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def update_reference(self, reference=None):
        if reference is not None:
            self.reference_keyframe = reference
        self.reference_constraint = (
            self.reference_keyframe.pose.inverse() * self.pose)

    def update_preceding(self, preceding=None):
        if preceding is not None:
            self.preceding_keyframe = preceding
        self.preceding_constraint = (
            self.preceding_keyframe.pose.inverse() * self.pose)

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed

    


class MapPoint(GraphMapPoint):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, normal, descriptor, 
            color=np.zeros(3), 
            covariance=np.identity(3) * 1e-4):
        super().__init__()

        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1

        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color

        self.count = defaultdict(int)

    def update_position(self, position):
        self.position = position
    def update_normal(self, normal):
        self.normal = normal
    def update_descriptor(self, descriptor):
        self.descriptor = descriptor
    def set_color(self, color):
        self.color = color

    def is_bad(self):
        with self._lock:
            status =  (
                self.count['meas'] == 0
                or (self.count['outlier'] > 20
                    and self.count['outlier'] > self.count['inlier'])
                or (self.count['proj'] > 50
                    and self.count['proj'] > self.count['meas'] * 50))
            return status

    def increase_outlier_count(self):
        with self._lock:
            self.count['outlier'] += 1
    def increase_inlier_count(self):
        with self._lock:
            self.count['inlier'] += 1
    def increase_projection_count(self):
        with self._lock:
            self.count['proj'] += 1
    def increase_measurement_count(self):
        with self._lock:
            self.count['meas'] += 1

    

class Measurement(GraphMeasurement):
    
    Source = Enum('Measurement.Source', ['TRIANGULATION', 'TRACKING', 'REFIND'])
    Type = Enum('Measurement.Type', ['STEREO', 'LEFT', 'RIGHT'])

    def __init__(self, type, source, keypoints, descriptors):
        super().__init__()

        self.type = type
        self.source = source
        self.keypoints = keypoints
        self.descriptors = descriptors

        self.xy = np.array(self.keypoints[0].pt)
        if self.is_stereo():
            self.xyx = np.array([
                *keypoints[0].pt, keypoints[1].pt[0]])

        self.triangulation = (source == self.Source.TRIANGULATION)

    def get_descriptor(self, i=0):
        return self.descriptors[i]
    def get_keypoint(self, i=0):
        return self.keypoints[i]

    def get_descriptors(self):
        return self.descriptors
    def get_keypoints(self):
        return self.keypoints

    def is_stereo(self):
        return self.type == Measurement.Type.STEREO
    def is_left(self):
        return self.type == Measurement.Type.LEFT
    def is_right(self):
        return self.type == Measurement.Type.RIGHT

    def from_triangulation(self):
        return self.triangulation
    def from_tracking(self):
        return self.source == Measurement.Source.TRACKING
    def from_refind(self):
        return self.source == Measurement.Source.REFIND