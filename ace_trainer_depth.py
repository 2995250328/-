# Copyright © Niantic, Inc. 2022.

import logging
import random
import time
from tqdm import *

import numpy as np
import math
import matplotlib
import cv2
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network_depth import Regressor,RelativeDepthLoss
from dataset import CamLocDataset
from superpoint import SuperPointFrontend
from depth_anything_v2.dpt import DepthAnythingV2

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)

def find_neighbors_with_confidence(coords, H, W, patch_size, include_diagonal=8):
    """
    Find neighbors for a 3D coordinate tensor with confidence values, filter duplicates
    based on confidence, and return the results.

    Args:
        coords (np.ndarray): Shape (N, 3), where each row contains (x, y, confidence).
        H (int): Image height.
        W (int): Image width.
        patch_size (int): Size of the patch.
        include_diagonal (bool): Whether to include diagonal neighbors.

    Returns:
        np.ndarray: Filtered neighbors with shape (M, 4), where each row contains
                    (x, y, confidence, original_index).
    """
    # Define neighbor offsets
    offsets =[(0,0)]
    if include_diagonal==4:
        offsets += [
            (-1, 0), (1, 0), (0, -1), (0, 1)  # Up, Down, Left, Right
        ]
    elif include_diagonal==8:
        offsets += [
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal neighbors
        ]

    # Initialize results storage
    neighbors = []
    #print(f'{H}   {W}')

    for idx, (x, y, confidence) in enumerate(coords):
        for dx, dy in offsets:
            nx, ny = x + dx * patch_size, y + dy * patch_size
            #print(f'{nx},{ny}')

            # Check if the neighbor is within bounds
            if 0 <= nx < W and 0 <= ny < H:
                neighbors.append((nx, ny, confidence, idx))
    # Convert to numpy array
    neighbors = np.array(neighbors, dtype=np.float32)
    #print(neighbors.shape)

    # Sort neighbors by coordinates and confidence (descending)
    neighbors = neighbors[np.lexsort((-neighbors[:, 2], neighbors[:, 1], neighbors[:, 0]))]

    # Remove duplicates by keeping the highest confidence
    unique_neighbors = []
    seen_coords = set()

    for x, y, conf, orig_idx in neighbors:
        coord_key = (x, y)
        if coord_key not in seen_coords:
            unique_neighbors.append((x, y, conf, orig_idx))
            seen_coords.add(coord_key)

    return np.array(unique_neighbors, dtype=np.float32)

def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE:
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda')
        self.cuda = self.device==torch.device('cuda')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.depth_iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.dataset = CamLocDataset(
            root_dir=self.options.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,    # Optional clustering for Cambridge experiments.
        )

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.options.scene,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous
        )
        # network_state_dict = torch.load(self.options.output_map_file, map_location="cpu")
        # self.regressor = Regressor.create_from_split_state_dict(encoder_state_dict, network_state_dict)
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")
        self.superpoint = SuperPointFrontend(
            self.regressor.superpoint,
            self.options.weights_path,
            self.options.nms_dist,
            self.options.conf_thresh,
            self.options.nn_thresh,self.cuda
            )
        
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.depth_anything = DepthAnythingV2(**model_configs[self.options.depth_encoder])
        self.depth_anything.load_state_dict(torch.load(f'depth_checkpoints/depth_anything_v2_{self.options.depth_encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)

        # Setup learning rate scheduler.
        steps_per_epoch = self.options.onebuffer // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)
        
        # 深度损失优化器
        # Setup optimization parameters.
        self.optimizer_depth = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        # Setup learning rate scheduler.
        self.scheduler_depth = optim.lr_scheduler.OneCycleLR(self.optimizer_depth,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.depth_epochs,
                                                       steps_per_epoch=self.options.depth_iterations,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)
        self.scaler_depth = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        #print(pixel_grid_2HW.shape)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )
        self.depth_loss = RelativeDepthLoss(self.options.RelativeDepthLoss_weight).to(self.device)

        # Will be filled at the beginning of the training process.
        self.training_buffer = None

        # Generate video of training process
        if self.options.render_visualization:
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.options.render_target_path,
                self.options.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.options.render_flipped_portrait,
                self.options.render_map_depth_filter,
                mapping_vis_error_threshold=self.options.render_map_error_threshold)
        else:
            self.ace_visualizer = None

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.options.render_camera_z_offset
            )

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()
        num_train = self.options.training_buffer_size//self.options.onebuffer

        #使用buffer训练
        for i in range(num_train):
            print(f'第{i+1}次buffer训练')
            # self.batch_generator.manual_seed(self.base_seed + 1023+i)
            # self.loader_generator.manual_seed(self.base_seed + 511+i)
            # self.sampling_generator.manual_seed(self.base_seed + 4095+i)
            # self.training_generator.manual_seed(self.base_seed + 8191+i)
            # Create training buffer.
            buffer_start_time = time.time()
            self.create_training_buffer()
            buffer_end_time = time.time()
            creating_buffer_time += buffer_end_time - buffer_start_time
            _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

            # Train the regression head.
            for self.epoch in range(self.options.epochs):
                epoch_start_time = time.time()
                self.run_epoch()
                training_time += time.time() - epoch_start_time
        
        #使用相对深度监督
        # Sampler.
        # batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
        #                                      batch_size=self.options.depth_batch_size,
        #                                      drop_last=False)
        # # Used to seed workers in a reproducible manner.
        # def seed_worker(worker_id):
        #     # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
        #     # the dataloader generator.
        #     worker_seed = torch.initial_seed() % 2 ** 32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)

        # training_dataloader = DataLoader(dataset=self.dataset,
        #                                  sampler=batch_sampler,
        #                                  batch_size=None,
        #                                  worker_init_fn=seed_worker,
        #                                  generator=self.loader_generator,
        #                                  pin_memory=True,
        #                                  num_workers=self.num_data_loader_workers,
        #                                  persistent_workers=self.num_data_loader_workers > 0,
        #                                  timeout=60 if self.num_data_loader_workers > 0 else 0,
        #                                  )       
        # for self.depth_epoch in range(self.options.depth_epochs):
        #     epoch_start_time = time.time()
        #     self.depth_run_epoch(training_dataloader)
        #     training_time += time.time() - epoch_start_time

        # Save trained model.
        self.save_model()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.options.scene / "train",
                mode=0,
                use_half=self.options.use_half,
                image_height=self.options.image_resolution,
                augment=False) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)

    def create_training_buffer_sp(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.onebuffer, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.onebuffer, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.onebuffer, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.options.onebuffer, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.options.onebuffer, 3, 3), dtype=torch.float32,
                                          device=self.device)
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0
            training_buffer_size = self.options.onebuffer
            
            while buffer_idx < self.options.onebuffer:
                dataset_passes += 1
                for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:
                    # print(image_B1HW.shape)
                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    _,_,I_H,I_W = image_B1HW.shape
                    #print(image_B1HW.shape)
                    img_rate = min(I_W,I_H)/480
                    #print(img_rate)
                    image_BHW = image_B1HW.squeeze(1)
                    image_HW = image_BHW.squeeze(0)
                    image_HW_np = image_HW.cpu().numpy().astype(np.float32)
                    #print(image_B1HW.shape)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    #print(image_mask_B1HW.shape)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)
                    pts, desc, _ = self.superpoint.run(image_HW_np)
                    if desc is None:
                        continue
                    #print(features_BCHW.shape)
                    # 得到特征点和描述符的tensor
                    pts_tensor = torch.from_numpy(pts)
                    desc_tensor = torch.from_numpy(desc)
                    pts_tensor = pts_tensor.permute(1,0).to(self.device,non_blocking=True,dtype=features_BCHW.dtype)
                    desc_tensor = desc_tensor.permute(1,0).to(self.device,non_blocking=True,dtype=features_BCHW.dtype)
                    #print(pts_tensor[0:3,:])
                    # 从superpoint提取得到特征点坐标（x，y），转换为所在patch的中心点坐标与pixel_positions_B2HW对应
                    pts_tensor[:,:2] = (pts_tensor[:,:2]//self.regressor.OUTPUT_SUBSAMPLE)*self.regressor.OUTPUT_SUBSAMPLE + \
                            torch.tensor(4,device=pts_tensor.device,dtype=pts_tensor.dtype)
                    #print(pts_tensor[0:3,:])                  

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape
                    #print(features_BCHW.shape)

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    # 每个grid坐标在生成时乘了下采样率，也就是说坐标对应的是原始图片的像素位置。
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    #print(pixel_positions_B2HW)
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    # N=B*H*W
                    gt_pose_inv = gt_pose_inv_B44[:, :3] 
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)
                    features_NC = normalize_shape(features_BCHW)
                    pixel_positions_N2 = normalize_shape(pixel_positions_B2HW)

                    # 从特征点对应patch的中心坐标出发，找相邻的8/4个patch中心坐标，得到对应坐标、置信度、来源坐标的索引
                    # 置信度是为了区分可能得到的相同patch坐标，来源索引是为了后续从desc向量中得到对应patch的描述符
                    pts_np = pts_tensor.cpu().numpy().astype(np.float32)
                    #print(pts_np.shape)
                    position_confidence_idx = find_neighbors_with_confidence(pts_np,I_H,I_W,self.regressor.OUTPUT_SUBSAMPLE,4)
                    position_confidence_idx = torch.from_numpy(position_confidence_idx).to(device=self.device,dtype=features_NC.dtype)
                    #print(position_confidence_idx.shape)
                    indices=[]
                    for coord in position_confidence_idx[:,:2]:
                        match = torch.all(pixel_positions_N2 == coord,dim=1)
                        idx = torch.where(match)[0]
                        if idx.numel()>0:
                            indices.append(idx.item())
                    indices = torch.tensor(indices,device=image_B1HW.device,dtype=torch.int)
                    #print(indices.shape)

                    # 从desc中得到对应坐标来源的描述符
                    desc_vector = []
                    for idx in position_confidence_idx[:,3]:
                        #print(idx)
                        if idx >= desc_tensor.shape[0]:
                            break
                        desc_vector.append(desc_tensor[idx.to(dtype=torch.int),:].tolist())
                    desc_vector = torch.tensor(desc_vector,device=image_B1HW.device,dtype=image_B1HW.dtype)
                    #print(desc_vector.shape)

                    batch_data = {
                        'features': features_NC,
                        'target_px': pixel_positions_N2,
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv
                    }

                    batch_data1 =batch_data.copy()
                    batch_data2 =batch_data.copy()

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    features_to_select = indices.shape[0]
                    #print(features_to_select)
                    features_to_select = torch.tensor(min(features_to_select, self.options.onebuffer - buffer_idx),
                                                    dtype=torch.int)   
                    for k in batch_data:
                        batch_data1[k] = batch_data[k][indices[:features_to_select]]       

                    # 采样数量小于预设值且未到结尾，补充采样         
                    supplement_flag = False    
                    supplement_select = 0   
                    samples_per_image = int(img_rate*self.options.samples_per_image * B)              
                    # print(f'每张图片采样数量{samples_per_image}')
                    if indices.shape[0] < samples_per_image and features_to_select == indices.shape[0]:
                        
                        supplement_select = samples_per_image - indices.shape[0]
                        supplement_select = torch.tensor(min(supplement_select, self.options.onebuffer - buffer_idx - features_to_select),
                                                    dtype=torch.int)   
                        if supplement_select != 0:
                            supplement_flag = True
                            sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                    supplement_select,
                                                    replacement=True,
                                                    generator=self.sampling_generator)
                            for k in batch_data:
                                batch_data2[k] = batch_data[k][sample_idxs]  
                               
                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select + supplement_select
                    # print(f'实际采样数量{features_to_select + supplement_select}')
                    if supplement_flag:
                        for k in batch_data:
                            self.training_buffer[k][buffer_idx:buffer_offset] = torch.cat((batch_data1[k],batch_data2[k]),dim=0)
                        # print("batch_data1 shape:", batch_data1['features'].shape)
                        # print("batch_data2 shape:", batch_data2['features'].shape)
                    else:
                        for k in batch_data:                          
                            self.training_buffer[k][buffer_idx:buffer_offset] = batch_data1[k]
                        # print("batch_data1 shape:", batch_data1['features'].shape)

                    buffer_idx = buffer_offset
                    print(f'\r{buffer_idx}/{self.options.onebuffer}',end="")
                    if buffer_idx >= self.options.onebuffer:
                        break
        print()
        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.onebuffer, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.onebuffer, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.onebuffer, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.options.onebuffer, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.options.onebuffer, 3, 3), dtype=torch.float32,
                                          device=self.device)
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            
            while buffer_idx < self.options.onebuffer:
                dataset_passes += 1
                for _,image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:

                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(features_to_select, self.options.onebuffer - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                    features_to_select,
                                                    replacement=True,
                                                    generator=self.sampling_generator)

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    print(f'\r{buffer_idx}/{self.options.onebuffer}',end="")
                    if buffer_idx >= self.options.onebuffer:
                        break
        # with torch.no_grad():
        #     # Iterate until the training buffer is full.
        #     buffer_idx = 0
        #     dataset_passes = 0
        #     training_buffer_size = self.options.onebuffer
            
        #     while buffer_idx < self.options.onebuffer:
        #         dataset_passes += 1
        #         for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:
        #             # print(image_B1HW.shape)
        #             # Copy to device.
        #             image_B1HW = image_B1HW.to(self.device, non_blocking=True)
        #             _,_,I_H,I_W = image_B1HW.shape
        #             #print(image_B1HW.shape)
        #             img_rate = min(I_W,I_H)/480
        #             #print(img_rate)
        #             image_BHW = image_B1HW.squeeze(1)
        #             image_HW = image_BHW.squeeze(0)
        #             image_HW_np = image_HW.cpu().numpy().astype(np.float32)
        #             #print(image_B1HW.shape)
        #             image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
        #             #print(image_mask_B1HW.shape)
        #             gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
        #             intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
        #             intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

        #             # Compute image features.
        #             with autocast(enabled=self.options.use_half):
        #                 features_BCHW = self.regressor.get_features(image_B1HW)
        #             pts, desc, _ = self.superpoint.run(image_HW_np)
        #             if desc is None:
        #                 continue
        #             #print(features_BCHW.shape)
        #             # 得到特征点和描述符的tensor
        #             pts_tensor = torch.from_numpy(pts)
        #             desc_tensor = torch.from_numpy(desc)
        #             pts_tensor = pts_tensor.permute(1,0).to(self.device,non_blocking=True,dtype=features_BCHW.dtype)
        #             desc_tensor = desc_tensor.permute(1,0).to(self.device,non_blocking=True,dtype=features_BCHW.dtype)
        #             #print(pts_tensor[0:3,:])
        #             # 从superpoint提取得到特征点坐标（x，y），转换为所在patch的中心点坐标与pixel_positions_B2HW对应
        #             pts_tensor[:,:2] = (pts_tensor[:,:2]//self.regressor.OUTPUT_SUBSAMPLE)*self.regressor.OUTPUT_SUBSAMPLE + \
        #                     torch.tensor(4,device=pts_tensor.device,dtype=pts_tensor.dtype)
        #             #print(pts_tensor[0:3,:])                  

        #             # Dimensions after the network's downsampling.
        #             B, C, H, W = features_BCHW.shape
        #             #print(features_BCHW.shape)

        #             # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
        #             image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
        #             image_mask_B1HW = image_mask_B1HW.bool()

        #             # If the current mask has no valid pixels, continue.
        #             if image_mask_B1HW.sum() == 0:
        #                 continue

        #             # Create a tensor with the pixel coordinates of every feature vector.
        #             # 每个grid坐标在生成时乘了下采样率，也就是说坐标对应的是原始图片的像素位置。
        #             pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
        #             #print(pixel_positions_B2HW)
        #             pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
        #             pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

        #             # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
        #             # N=B*H*W
        #             gt_pose_inv = gt_pose_inv_B44[:, :3] 
        #             gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

        #             # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
        #             intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
        #             intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

        #             def normalize_shape(tensor_in):
        #                 """Bring tensor from shape BxCxHxW to NxC"""
        #                 return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)
        #             features_NC = normalize_shape(features_BCHW)
        #             pixel_positions_N2 = normalize_shape(pixel_positions_B2HW)

        #             # 从特征点对应patch的中心坐标出发，找相邻的8/4个patch中心坐标，得到对应坐标、置信度、来源坐标的索引
        #             # 置信度是为了区分可能得到的相同patch坐标，来源索引是为了后续从desc向量中得到对应patch的描述符
        #             pts_np = pts_tensor.cpu().numpy().astype(np.float32)
        #             #print(pts_np.shape)
        #             position_confidence_idx = find_neighbors_with_confidence(pts_np,I_H,I_W,self.regressor.OUTPUT_SUBSAMPLE,4)
        #             position_confidence_idx = torch.from_numpy(position_confidence_idx).to(device=self.device,dtype=features_NC.dtype)
        #             #print(position_confidence_idx.shape)
        #             indices=[]
        #             for coord in position_confidence_idx[:,:2]:
        #                 match = torch.all(pixel_positions_N2 == coord,dim=1)
        #                 idx = torch.where(match)[0]
        #                 if idx.numel()>0:
        #                     indices.append(idx.item())
        #             indices = torch.tensor(indices,device=image_B1HW.device,dtype=torch.int)
        #             #print(indices.shape)

        #             # 从desc中得到对应坐标来源的描述符
        #             desc_vector = []
        #             for idx in position_confidence_idx[:,3]:
        #                 #print(idx)
        #                 if idx >= desc_tensor.shape[0]:
        #                     break
        #                 desc_vector.append(desc_tensor[idx.to(dtype=torch.int),:].tolist())
        #             desc_vector = torch.tensor(desc_vector,device=image_B1HW.device,dtype=image_B1HW.dtype)
        #             #print(desc_vector.shape)

        #             batch_data = {
        #                 'features': features_NC,
        #                 'target_px': pixel_positions_N2,
        #                 'gt_poses_inv': gt_pose_inv,
        #                 'intrinsics': intrinsics,
        #                 'intrinsics_inv': intrinsics_inv
        #             }

        #             batch_data1 =batch_data.copy()
        #             batch_data2 =batch_data.copy()

        #             # Turn image mask into sampling weights (all equal).
        #             image_mask_B1HW = image_mask_B1HW.float()
        #             image_mask_N1 = normalize_shape(image_mask_B1HW)

        #             features_to_select = indices.shape[0]
        #             #print(features_to_select)
        #             features_to_select = torch.tensor(min(features_to_select, self.options.onebuffer - buffer_idx),
        #                                             dtype=torch.int)   
        #             for k in batch_data:
        #                 batch_data1[k] = batch_data[k][indices[:features_to_select]]       

        #             # 采样数量小于预设值且未到结尾，补充采样         
        #             supplement_flag = False    
        #             supplement_select = 0   
        #             samples_per_image = int(img_rate*self.options.samples_per_image * B)              
        #             # print(f'每张图片采样数量{samples_per_image}')
        #             if indices.shape[0] < samples_per_image and features_to_select == indices.shape[0]:
                        
        #                 supplement_select = samples_per_image - indices.shape[0]
        #                 supplement_select = torch.tensor(min(supplement_select, self.options.onebuffer - buffer_idx - features_to_select),
        #                                             dtype=torch.int)   
        #                 if supplement_select != 0:
        #                     supplement_flag = True
        #                     sample_idxs = torch.multinomial(image_mask_N1.view(-1),
        #                                             supplement_select,
        #                                             replacement=True,
        #                                             generator=self.sampling_generator)
        #                     for k in batch_data:
        #                         batch_data2[k] = batch_data[k][sample_idxs]  
                               
        #             # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
        #             buffer_offset = buffer_idx + features_to_select + supplement_select
        #             # print(f'实际采样数量{features_to_select + supplement_select}')
        #             if supplement_flag:
        #                 for k in batch_data:
        #                     self.training_buffer[k][buffer_idx:buffer_offset] = torch.cat((batch_data1[k],batch_data2[k]),dim=0)
        #                 # print("batch_data1 shape:", batch_data1['features'].shape)
        #                 # print("batch_data2 shape:", batch_data2['features'].shape)
        #             else:
        #                 for k in batch_data:                          
        #                     self.training_buffer[k][buffer_idx:buffer_offset] = batch_data1[k]
        #                 # print("batch_data1 shape:", batch_data1['features'].shape)

        #             buffer_idx = buffer_offset
        #             print(f'\r{buffer_idx}/{self.options.onebuffer}',end="")
        #             if buffer_idx >= self.options.onebuffer:
        #                 break
        print()
        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.options.onebuffer, generator=self.training_generator)

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=self.options.depth_batch_size,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        # Iterate with mini batches.
        for batch_start in range(0, self.options.onebuffer, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > self.options.onebuffer:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous(),
                self.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous(),
                training_dataloader
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33,training_dataloader):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        # 实际上是把channels拆成了16*32
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        #print(features_bCHW.shape)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size      
        # 可调节深度损失进入时间
        if self.iteration>0.25*self.iterations:
            loss_depth = self.depth_run_epoch(training_dataloader)
            loss = loss + (self.iteration/(self.iterations*(1/self.options.DepthLoss_weight)))*loss_depth

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def RGBD_run_epoch(self,training_dataloader):
        return 0

    def depth_run_epoch(self,training_dataloader):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True
        # # Sampler.
        # batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
        #                                      batch_size=self.options.depth_batch_size,
        #                                      drop_last=False)

        # # Used to seed workers in a reproducible manner.
        # def seed_worker(worker_id):
        #     # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
        #     # the dataloader generator.
        #     worker_seed = torch.initial_seed() % 2 ** 32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)

        # training_dataloader = DataLoader(dataset=self.dataset,
        #                                  sampler=batch_sampler,
        #                                  batch_size=None,
        #                                  worker_init_fn=seed_worker,
        #                                  generator=self.loader_generator,
        #                                  pin_memory=True,
        #                                  num_workers=self.num_data_loader_workers,
        #                                  persistent_workers=self.num_data_loader_workers > 0,
        #                                  timeout=60 if self.num_data_loader_workers > 0 else 0,
        #                                  )
       
        # for image_RGB,image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:
        batch_data = next(iter(training_dataloader))
        image_RGB, image_B1HW, image_mask_B1HW, gt_pose_B44,gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ = batch_data
        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
        gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
        intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
        intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)
        image_RGB = image_RGB.permute(0, 2, 3, 1).squeeze(0).numpy().astype(np.uint8)  # 调整维度顺序为 HWC
        image_BGR = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)  # RGB→BGR转换
        with autocast(enabled=self.options.use_half):
            with torch.no_grad():
                features_BCHW = self.regressor.get_features(image_B1HW)
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_BCHW)
        B, C, H, W = features_BCHW.shape
        with torch.no_grad():
            depth_HW = self.depth_anything.infer_image(image_BGR,self.options.input_size)
        # 归一化深度图
        depth_HW = (depth_HW - depth_HW.min()) / (depth_HW.max() - depth_HW.min())
        depth_HW = torch.from_numpy(depth_HW).to(self.device,dtype=pred_scene_coords_b3HW.dtype).unsqueeze(0).unsqueeze(0)
        depth_HW = F.avg_pool2d(depth_HW,kernel_size=8, stride=8).squeeze(0).squeeze(0)      

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_N31 = pred_scene_coords_b3HW[:,:,:depth_HW.shape[0],:depth_HW.shape[1]].permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_N41 = to_homogeneous(pred_scene_coords_N31)

        gt_pose_inv = gt_pose_inv_B44[:, :3]
        gt_inv_poses_b34 = gt_pose_inv.unsqueeze(1).expand(B, depth_HW.shape[0] * depth_HW.shape[1], 3, 4).reshape(-1, 3, 4)
        # Scene coordinates to camera coordinates.
        pred_cam_coords_N31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_N41)            
        pred_cam_coords_b3HW = pred_cam_coords_N31.squeeze(-1).unflatten(0,(B,depth_HW.shape[0],depth_HW.shape[1])).permute(0, 3, 1, 2)

        # Project scene coordinates.
        intrinsics_B33 = intrinsics_B33.repeat(pred_cam_coords_N31.shape[0],1,1)
        pred_px_b31 = torch.bmm(intrinsics_B33, pred_cam_coords_N31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        # Create a tensor with the pixel coordinates of every feature vector.
        pixel_positions_B2HW = self.pixel_grid_2HW[:, :depth_HW.shape[0],:depth_HW.shape[1]].clone()  # It's 2xHxW (actual H and W) now.
        pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
        pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, depth_HW.shape[0], depth_HW.shape[1])  # Bx2xHxW
        def normalize_shape(tensor_in):
                """Bring tensor from shape BxCxHxW to NxC"""
                return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

        target_px_b2 = normalize_shape(pixel_positions_B2HW)
        reprojection_error_b2 = pred_px_b21.squeeze(2) - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)
        reprojection_error_B1HW = reprojection_error_b1.unflatten(0,(B,depth_HW.shape[0],depth_HW.shape[1])).permute(0, 3, 1, 2)

        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b3HW[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_B1HW > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b3HW[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_B1HW = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_B1HW = ~invalid_mask_B1HW
        valid_mask_HW = valid_mask_B1HW.squeeze(0).squeeze(0)

        pred_depth = pred_cam_coords_b3HW[:,2,:,:].squeeze(0)
        pred_depth = 1-(pred_depth-torch.min(pred_depth)) / (torch.max(pred_depth)-torch.min(pred_depth)+1e-5)
        # 尺度对齐
        # s = torch.median(depth_HW) / torch.median(pred_depth)
        # pred_depth = pred_depth*s
        # 深度图可视化
        # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        # pred_depth_np = pred_depth*255.0
        # pred_depth_np = pred_depth_np.detach().cpu().numpy().astype(np.uint8)
        # pred_depth_np = (cmap(pred_depth_np)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        # depth_HW_np = depth_HW*255.0
        # depth_HW_np = depth_HW_np.detach().cpu().numpy().astype(np.uint8)
        # depth_HW_np = (cmap(depth_HW_np)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        # cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
        # combined = np.hstack((pred_depth_np,depth_HW_np))
        # # 将拼接后的图像放大 2 倍
        # display_img = cv2.resize(combined, (combined.shape[1] * 4, combined.shape[0] * 4))
        # cv2.resizeWindow("Combined Image", combined.shape[1] * 4, combined.shape[0] * 4)
        # cv2.imshow("Combined Image", display_img)
        # cv2.imshow("Raw Image", image_BGR)
        # key = cv2.waitKey(0)
        # if key == 27:  
        #     cv2.destroyAllWindows()

        loss = self.depth_loss(pred_depth,depth_HW,valid_mask_HW)
        if math.isnan(loss):
            print('pred_depth')
            print(pred_depth)
            print('depth_HW')
            print(depth_HW)
            print('invalid_mask_HW')
            print(invalid_mask_B1HW.sum())
            return 0
        return loss  
        # # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        # old_optimizer_step = self.optimizer_depth._step_count

        # # Optimization steps.
        # self.optimizer_depth.zero_grad(set_to_none=True)
        # self.scaler_depth.scale(loss).backward()
        # self.scaler_depth.step(self.optimizer_depth)
        # self.scaler_depth.update()

        # if self.depth_iteration % 20 == 0:
        #     # Print status.
        #     time_since_start = time.time() - self.training_start   
        #     fraction_valid = float(valid_mask_HW.sum() / (depth_HW.shape[0]*depth_HW.shape[1]))                

        #     _logger.info(f'Iteration: {self.depth_iteration:6d} / Epoch {self.depth_epoch:03d}|{self.options.depth_epochs:03d}, '
        #                 f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}% ,Time: {time_since_start:.2f}s')

        # self.depth_iteration += 1
        # # if self.depth_iteration == self.options.depth_iterations:
        # #     break

        # # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        # if old_optimizer_step < self.optimizer_depth._step_count < self.scheduler_depth.total_steps:
        #     self.scheduler_depth.step()
            
        
        
        # print('11111111111111111111111111111111')
        # print(pred_depth_valid)
        # print('22222222222222222222222222')
        # print(depth_HW_valid)






        




    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        model_state_dict = {}

        # Save the parameters of each module in half-precision
        model_state_dict["superpoint"] = {k: v.half() for k, v in self.regressor.superpoint.state_dict().items()}
        model_state_dict["heads"] = {k: v.half() for k, v in self.regressor.heads.state_dict().items()}

        # Save the dictionary to a file

        torch.save(model_state_dict, self.options.output_map_depth)

        _logger.info(f"Saved trained weights for superpoint and heads to: {self.options.output_map_depth}")