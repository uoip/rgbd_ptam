import cv2



class Params(object):
    def __init__(self):
        self.feature_detector = cv2.GFTTDetector_create(
            maxCorners=600, minDistance=15.0, 
            qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
            bytes=32, use_orientation=False)
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 2
        self.matching_distance = 30
        self.matching_distance_ratio = 0.8

        self.virtual_baseline = 0.1  # meters
        self.depth_near = 0.1
        self.depth_far = 10
        self.frustum_near = 0.1 
        self.frustum_far = 50.0
        
        self.pnp_min_measurements = 30
        self.pnp_max_iterations = 10
        self.init_min_points = 30

        self.local_window_size = 10
        self.keyframes_buffer_size = 5
        self.ba_max_iterations = 10

        self.min_tracked_points = 150
        self.min_tracked_points_ratio = 0.75

        self.lc_min_inbetween_keyframes = 15   # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 30
        self.lc_inliers_threshold = 13
        self.lc_inliers_ratio = 0.3
        self.lc_distance_threshold = 1.5      # meters
        self.lc_max_iterations = 20

        self.view_camera_width = 0.05
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000
        self.view_image_width = 400
        self.view_image_height = 250

    def relax_tracking(self, relax=True):
        if relax:
            self.matching_neighborhood = 5
        else:
            self.matching_neighborhood = 2