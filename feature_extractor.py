import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
import torch.functional as F

from models.blip_itm import blip_itm

import ffmpeg
import random
from tqdm import tqdm
import argparse
from pathlib import Path
import os
import pickle
import clip


class VideoLoader(Dataset):
    """Pytorch video loader."""
    def __init__(self, video_path, stride=8, fps=None, size=224, centercrop=True, ):
        self.centercrop = centercrop
        self.size = size
        self.stride = stride
        self.fps = fps
        self.frames = self._get_video_frames(video_path)

    def __len__(self):
        return len(self.frames)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = eval(video_stream['avg_frame_rate'])
        return height, width, fps

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def _get_video_frames(self, video_path):
        if os.path.isfile(video_path):
            # print('Decoding video: {}'.format(video_path))
            try:
                h, w, fps = self._get_video_dim(video_path)
            except:
                print('ffprobe failed at: {}'.format(video_path))
                return torch.zeros(1)
            height, width = self._get_output_dim(h, w)
            cmd = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=fps/self.stride if self.fps is None else self.fps)
                .filter('scale', width, height)
            )
            if self.centercrop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = torch.from_numpy(video.astype('float32'))
            video = video.permute(0, 3, 1, 2)
        else:
            print('file not find: {}'.format(video_path))
            video = torch.zeros(1)
            
        return video

    def __getitem__(self, idx):
        return self.frames[idx]


def extract_video_features(args, video_path, save_path, model, process, num_sntc=10):
    dataset = VideoLoader(video_path, args.stride, args.fps, args.input_size)
    if len(dataset.frames.shape) != 4:
        return
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    features = []
    with torch.no_grad():
        for data in dataloader:
            data = process(data/255.).cuda()
            image_embeds = model.visual_encoder(data)
            image_feat = model.vision_proj(image_embeds[:,0,:])
            features.append(image_feat.cpu().numpy())
    features = np.concatenate(features, axis=0)
    with open(save_path, 'wb') as f:
        np.save(f, features)


def extract_text_features(args, video_path, save_path, model):
    with open(video_path, 'rb') as f:
        data = pickle.load(f)
    num_stnc = len(data[0])
    assert all(len(d) == num_stnc for d in data)
    data = [s for d in data for s in d ]
    
    with torch.no_grad():
        features = []
        for i in range(0, len(data), args.batch_size):
            if args.model == 'blip':
                text = model.tokenizer(data[i:i+args.batch_size], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
                text_output = model.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')                     
                text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
            else:
                text = clip.tokenize(data[i:i+args.batch_size], truncate=True).to('cuda')
                text_feat = model.encode_text(text)
            features.append(text_feat)
        features = torch.cat(features, dim=0).view(-1, num_stnc, text_feat.size(-1)).cpu().numpy()

    with open(save_path, 'wb') as f:
        np.save(f, features)


def get_args():
    parser = argparse.ArgumentParser(description='Extract BLIP features for videos')
    parser.add_argument('--input_root', default='/home/zhengmh/Datasets/Charades/Charades_v1_480/', type=str)
    parser.add_argument('--save_root', default='/home/zhengmh/Datasets/Charades/blip_features/', type=str)
    parser.add_argument('--model', default='blip', choices=['blip', 'clip'])
    parser.add_argument('--stride', default=8, type=int)
    parser.add_argument('--fps', default=None, type=float)
    parser.add_argument('--input_size', default=384, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--extract_text', action='store_true')

    return parser.parse_args()

def main(args):
    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    if args.model == 'blip':
        model = blip_itm(pretrained='checkpoints/model_base_retrieval_coco.pth', image_size=384, vit='base').cuda()
        process = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        model, preprocess = clip.load("ViT-B/32")
        model = model.float()

    list_dir = os.listdir(args.input_root)
    list_dir = random.sample(list_dir, len(list_dir))
    # list_dir = ['5UNDJ.mp4']
    for video_name in tqdm(list_dir):
        vid = video_name.split('.')[0]
        input_path = os.path.join(args.input_root, video_name)
        save_path = os.path.join(args.save_root, vid+'.npy')
        if os.path.isfile(save_path) or not os.path.isfile(input_path):
            continue
        if args.extract_text:
            extract_text_features(args, input_path, save_path, model)
        else:
            extract_video_features(args, input_path, save_path, model, process)

        
if __name__=='__main__':
    main(get_args())
