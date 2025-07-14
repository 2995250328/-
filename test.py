import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision import transforms
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dataset import CamLocDataset
from pathlib import Path
from torch.utils.data import DataLoader

# 直接使用固定参数初始化数据集
dataset = CamLocDataset(
    root_dir=Path("datasets/pgt_7scenes_pumpkin") / "train",
    mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
    use_half=True,
    image_height=480,
    augment=True,
    aug_rotation=15,
    aug_scale_max=1.5,
    aug_scale_min=1 / 1.5,
    num_clusters=None,  # Optional clustering for Cambridge experiments.
    cluster_idx=None,    # Optional clustering for Cambridge experiments.
)

# 创建数据加载器
dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

def process_image(image, model_size="large"):
    """处理图像并提取深度图"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保图像尺寸是14的倍数
    h, w = image.shape[-2:]  # 获取图像的高度和宽度
    new_h = ((h + 13) // 14) * 14  # 向上取整到最近的14的倍数
    new_w = ((w + 13) // 14) * 14  # 向上取整到最近的14的倍数
    
    # 图像预处理 - 转换为float并归一化
    if image.dtype != torch.float32:
        image = image.float()
    
    # 如果图像值在0-255范围内，归一化到0-1
    if image.max() > 1.0:
        print('大于')
        image = image / 255.0
    
    image_tensor = image.to(device)
    print(image_tensor.shape)
    
    # 模型配置
    model_kwargs = {
        "large": dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        ),
        "base": dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        "small": dict(
            encoder='vits',
            features=64,
            out_channels=[48, 96, 192, 384],
        )
    }
    
    # 加载模型
    checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"small/model.safetensors", repo_type="model")
    
    if model_size == "large":
        model = DepthAnything(**model_kwargs[model_size]).to(device)
    else:
        model = DepthAnythingV2(**model_kwargs[model_size]).to(device)
    
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model.eval()
    
    # 推理
    with torch.no_grad():
        pred_disp, _ = model(image_tensor)
    
    # 处理深度图
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
    
    # 着色
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    
    # 调整尺寸回原始尺寸
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    
    # 确保返回的图像格式正确
    original_image = image.cpu().numpy()
    if len(original_image.shape) == 4:  # 如果是4D (B, C, H, W)
        original_image = original_image[0]  # 取第一个batch
    if original_image.shape[0] == 3:  # CHW格式
        original_image = np.transpose(original_image, (1, 2, 0))  # 转换为HWC格式
    
    return original_image, depth_colored_hwc

def resize_to_multiple_of_14(tensor):
    """调整tensor尺寸到14的倍数"""
    if tensor.dim() == 4:  # (B, C, H, W)
        _, _, h, w = tensor.shape
    elif tensor.dim() == 3:  # (C, H, W)
        _, h, w = tensor.shape
    else:
        return tensor
    
    new_h = ((h + 13) // 14) * 14  # 向上取整到最近的14的倍数
    new_w = ((w + 13) // 14) * 14  # 向上取整到最近的14的倍数
    
    if new_h != h or new_w != w:
        # 确保输入是4D tensor (N, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # 添加batch维度
        
        # 根据数据类型选择插值模式
        if tensor.dtype == torch.bool:
            # 布尔类型使用nearest插值
            tensor = torch.nn.functional.interpolate(
                tensor.float(),  # 转换为float进行插值
                size=(new_h, new_w), 
                mode='nearest'
            ).bool()  # 转换回bool
        else:
            # 其他类型使用bilinear插值
            tensor = torch.nn.functional.interpolate(
                tensor, 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 如果原来是3D，去掉batch维度
        if tensor.shape[0] == 1 and tensor.dim() == 3:
            tensor = tensor.squeeze(0)
    
    return tensor

def show_depth_map_from_dataloader(model_size="large"):
    """从数据加载器中读取图片并显示深度图"""
    try:
        # 从数据加载器中获取第一张图片
        for data in dataloader:
            # 先检查数据格式
            print(f"数据类型: {type(data)}")
            if isinstance(data, (list, tuple)):
                print(f"数据长度: {len(data)}")
                for i, item in enumerate(data):
                    print(f"项目 {i}: {type(item)}, 形状: {item.shape if hasattr(item, 'shape') else 'N/A'}")
            
            # 根据实际数据长度处理
            if isinstance(data, (list, tuple)):
                if len(data) >= 9:
                    # 完整格式：9个元素
                    image_RGB, image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ = data
                elif len(data) >= 3:
                    # 简化格式：至少3个元素
                    image_RGB, image_B1HW, image_mask_B1HW = data[:3]
                    gt_pose_B44 = gt_pose_inv_B44 = intrinsics_B33 = intrinsics_inv_B33 = None
                elif len(data) >= 2:
                    # 更简化格式：至少2个元素
                    image_RGB, image_B1HW = data[:2]
                    image_mask_B1HW = None
                    gt_pose_B44 = gt_pose_inv_B44 = intrinsics_B33 = intrinsics_inv_B33 = None
                else:
                    # 单个元素
                    image_RGB = data[0]
                    image_B1HW = image_RGB
                    image_mask_B1HW = None
                    gt_pose_B44 = gt_pose_inv_B44 = intrinsics_B33 = intrinsics_inv_B33 = None
            else:
                # 直接是图像数据
                image_RGB = data
                image_B1HW = data
                image_mask_B1HW = None
                gt_pose_B44 = gt_pose_inv_B44 = intrinsics_B33 = intrinsics_inv_B33 = None
            
            # 调整所有图像数据的尺寸到14的倍数（如果存在）
            image_RGB = resize_to_multiple_of_14(image_RGB)
            image_B1HW = resize_to_multiple_of_14(image_B1HW)
            if image_mask_B1HW is not None:
                image_mask_B1HW = resize_to_multiple_of_14(image_mask_B1HW)
            
            # 只处理第一张图片
            image = image_B1HW[0] if image_B1HW.dim() == 4 else image_B1HW
            
            # 处理图像并获取深度图
            print(image_RGB.shape)
            image_np, depth_map = process_image(image_RGB, model_size)
            
            # 显示结果
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 确保图像格式正确用于显示
            if len(image_np.shape) == 4:  # 如果是4D (B, C, H, W)
                image_display = image_np[0]  # 取第一个batch
                image_display = np.transpose(image_display, (1, 2, 0))  # 转换为HWC格式
            elif image_np.shape[0] == 3:  # 如果是CHW格式
                image_display = np.transpose(image_np, (1, 2, 0))
            else:
                image_display = image_np
            
            # 确保值在0-1范围内
            if image_display.max() > 1.0:
                image_display = image_display / 255.0
            
            ax1.imshow(image_display)
            ax1.set_title('原图 (从数据加载器读取)')
            ax1.axis('off')
            
            ax2.imshow(depth_map)
            ax2.set_title(f'深度图 ({model_size} 模型)')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # 只处理一次，然后退出循环
            break
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        print(f"数据集长度: {len(dataset)}")
        # 尝试直接访问数据集
        try:
            sample = dataset[0]
            print(f"数据集样本类型: {type(sample)}")
            if isinstance(sample, (list, tuple)):
                print(f"样本长度: {len(sample)}")
        except Exception as e2:
            print(f"访问数据集样本时出错: {e2}")

if __name__ == "__main__":
    # 从数据加载器中读取图片并显示深度图
    show_depth_map_from_dataloader(model_size="small")