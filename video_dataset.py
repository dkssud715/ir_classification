import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any


class VideoRecord(object):
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    def path(self) -> str:
        return self._path
    
    def num_frames(self) -> int:
        return self.end_frame() - self.start_frame() + 1  

    def start_frame(self) -> int:
        return int(self._data[1])
    
    def end_frame(self) -> int:
        return int(self._data[2])

    def label(self) -> Union[int, List[int]]:
        if len(self._data) == 4:
            return int(self._data[3])
        else:
            return [int(label_id) for label_id in self._data[3:]]

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 1,
                 frames_per_segment: int = 5,
                 imagefile_template: str='img_{:05d}.jpg',
                 transform = None,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert("L")

    def _parse_annotationfile(self):
        self.video_list = [VideoRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames() <= 0 or record.start_frame() == record.end_frame():
                print(f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

            elif record.num_frames() < (self.num_segments * self.frames_per_segment):
                print(f"\nDataset Warning: video {record.path()} has {record.num_frames()} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        if self.test_mode:
            distance_between_indices = (record.num_frames() - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        else:
            max_valid_start_index = (record.num_frames() - self.frames_per_segment + 1) // self.num_segments

            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                      np.random.randint(max_valid_start_index, size=self.num_segments)
            #print(start_indices)
        return start_indices

    def __getitem__(self, idx: int) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
        ]:

        record: VideoRecord = self.video_list[idx]

        frame_start_indices: 'np.ndarray[int]' = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]') -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
        ]:
        frame_start_indices = frame_start_indices + record.start_frame()
        images = list()

        for start_index in frame_start_indices:
            frame_index = int(start_index)

            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path(), frame_index)
                images.append(image)

                if frame_index < record.end_frame():
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label()

    def __len__(self):
        return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        return torch.stack([transforms.functional.to_tensor(pic) / 255.0 for pic in img_list])