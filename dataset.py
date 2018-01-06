import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue



class ImageReader(object):
    def __init__(self, ids, timestamps=None, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype
    @property
    def shape(self):
        return self[0].shape




class ICLNUIMDataset(object):
    '''
    path example: 'path/to/your/ICL-NUIM R-GBD Dataset/living_room_traj0_frei_png'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.rgb = ImageReader(self.listdir(os.path.join(path, 'rgb')))
        self.depth = ImageReader(self.listdir(os.path.join(path, 'depth')))
        self.timestamps = None

    def sort(self, xs):
        return sorted(xs, key=lambda x:int(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.rgb)




def make_pair(matrix, threshold=1):
    assert (matrix >= 0).all()
    pairs = []
    base = defaultdict(int)
    while True:
        i = matrix[:, 0].argmin()
        min0 = matrix[i, 0]
        j = matrix[0, :].argmin()
        min1 = matrix[0, j]

        if min0 < min1:
            i, j = i, 0
        else:
            i, j = 0, j
        if min(min1, min0) < threshold:
            pairs.append((i + base['i'], j + base['j']))

        matrix = matrix[i + 1:, j + 1:]
        base['i'] += (i + 1)
        base['j'] += (j + 1)

        if min(matrix.shape) == 0:
            break
    return pairs


class TUMRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        525.0, 525.0, 319.5, 239.5, 5000)
    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        for name in self.sort(files):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)