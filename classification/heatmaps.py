import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

class GradCAM():
    def __init__(self, config, model, target_layer = None, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """
        Args:
            model: 模型
            target_layer: 目标卷积层（通常是最后一个卷积层）
        """
        self.config = config
        self.model = model.to(device)
        self.model.eval()
        if hasattr(model.resnet, 'model'):
        # 如果是timm模型
            self.target_layer = model.resnet.model.layer3[-1].conv2
        else:
        # 标准ResNet结构
            self.target_layer = model.resnet.layer3[-1].conv2
        # self.target_layer = target_layer

        self.device = device
        self.gradients = None
        self.activations = None
        
        # 注册钩子来获取激活和梯度
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 注册钩子[3,5](@ref)
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx = None):
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像张量 [1, 3, H, W]
            class_idx: 目标类别索引，如果为None则使用预测类别
        """
        # 设置为评估模式
        self.model.eval()
        
        # 前向传播
        # output = self.model(input_tensor)
        pred, scores = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = pred.item() if pred.dim() == 0 else pred[0].item()

        # 创建one-hot向量用于反向传播[5](@ref)
        one_hot = torch.zeros_like(scores)
        one_hot[0, class_idx] = 1

        # 清零梯度
        self.model.zero_grad()
        
        # 反向传播计算梯度
        scores.backward(gradient = one_hot, retain_graph = True)
        
        # 检查是否成功获取梯度和激活
        if self.gradients is None or self.activations is None:
            raise RuntimeError("未能获取梯度或激活，请检查钩子注册")
        
        # 计算权重：对梯度在空间维度上求平均[1](@ref)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 计算加权激活
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # 应用ReLU去除负值
        
        # 归一化到[0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 在返回前确保热力图是2D的
        if cam.dim() > 2:
            cam = cam.squeeze()
        
        # 确保是numpy数组且是2D
        cam_np = cam.cpu().numpy() if torch.is_tensor(cam) else cam
        if cam_np.ndim > 2:
            cam_np = cam_np[0]  # 取第一个通道（如果是多通道）
        
        return cam_np, class_idx

    def visualize_gradcam(self, original_image, heatmap, alpha = 0.5):  # 必须添加self参数，否则original_image自动变成GradCAM类的属性
        """
        可视化Grad-CAM结果
        
        Args:
            original_image: 原始图像 [H, W, 3]
            heatmap: 热力图 [H, W]
            alpha: 热力图透明度
        """

        # print(type(original_image))

        # 调整热力图大小匹配原图
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # 应用颜色映射（蓝-青-黄-红）[1](@ref)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加热力图到原图
        superimposed_img = heatmap_colored * alpha + original_image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img, heatmap_colored

# 针对你的模型的完整可视化流程
    def create_gradcam_for_model(self, image_tensor, original_image = None, labels = None):
        """
        为你的Univision模型创建Grad-CAM可视化
        
        Args:
            model: 你的Univision模型实例
            image_tensor: 预处理后的图像张量 [1, 3, H, W]
            original_image: 原始图像用于显示 [H, W, 3]
        """
        
        # 生成CAM
        heatmap, pred_class = self.generate_cam(image_tensor, labels)
        
        # 可视化
        if original_image is not None:
            # 如果提供了原始图像，进行叠加可视化
            superimposed_img, colored_heatmap = self.visualize_gradcam(original_image, heatmap)
            
            # 绘制结果
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 热力图
            axes[1].imshow(colored_heatmap)
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # 叠加图像
            axes[2].imshow(superimposed_img)
            axes[2].set_title(f'Overlay (Class: {pred_class})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.savefig('/mnt/breast_cancer_multimodal/model/1_Grad-CAM.png')
        
        return heatmap, pred_class


# 假设你已经有了训练好的模型和测试数据
    def test_gradcam_on_single_image(self, test_dataloader):
        """在单个测试图像上应用Grad-CAM"""
        
        # 获取一个测试batch
        for batch in test_dataloader:
            guids, images, labels = batch
            image_tensor = images[0: 1].to(self.device)  # 取第一个图像 [1, 3, H, W]
            original_image = images[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            
            # print(type(original_image))

            # 反归一化原始图像用于显示
            original_image = (original_image * np.array([0.229, 0.224, 0.226]) + 
                            np.array([0.485, 0.456, 0.406]))
            original_image = np.clip(original_image, 0, 1)
            
            # 应用Grad-CAM
            heatmap, pred_class = self.create_gradcam_for_model(image_tensor, original_image)
            
            break  # 只处理第一个图像

# 批量处理多个图像
    def batch_gradcam_visualization(self, dataloader, patch_size = 256, scale_factor = 0.1, blur_sigma = 3):
        """批量生成Grad-CAM可视化"""

        # 创建缩放的画布
        heatmap_w = int(2048 * scale_factor)
        heatmap_h = int(1536 * scale_factor)
        # wsi_heatmap = np.zeros((heatmap_h, heatmap_w), dtype = np.float32)
        # count_map = np.zeros((heatmap_h, heatmap_w), dtype = np.int32)

        all_attention_scores = []
        all_coords = []
        
        for batch in tqdm(dataloader, desc = 'Grad-CAM Visualization-------------'):
            if self.config.model_type == 'multimodal':
                guids, imgs_list, coords_list, _, _, labels = batch
            else:
                guids, imgs_list, coords_list, labels = batch
            
            for i, (guid, patch_imgs, patch_coords, label) in enumerate(zip(guids, imgs_list, coords_list, labels)):
                patch_attention_scores = []
                
                wsi_heatmap = np.zeros((heatmap_h, heatmap_w), dtype = np.float32)
                count_map = np.zeros((heatmap_h, heatmap_w), dtype = np.int32)

                #根据guid和label读取tif文件，获取缩略图
                if label == 0:
                    wsi_path = '/mnt/Data/breast_cancer/image/' + 'benign_' + guid + '.tif'
                else:
                    wsi_path = '/mnt/Data/breast_cancer/image/' + 'malignant_' + guid + '.tif'
                thumbnail = cv2.imread(wsi_path)

                # 处理该样本的所有patch
                for j, (patch_img, coord) in enumerate(zip(patch_imgs, patch_coords)):
                    
                    # img_tensor = patch_img.to(self.device)
                    img_tensor = patch_imgs[j:j+1].to(self.device)


                    score, _ = self.generate_cam(img_tensor, label)
                    cam_resized = cv2.resize(score, (patch_size, patch_size))
                    target_region = wsi_heatmap[y_scaled:y_end, x_scaled:x_end]
            
                    # 确保形状匹配
                    if cam_resized.shape != target_region.shape:
                        cam_final = cv2.resize(cam_resized, 
                                            (target_region.shape[1], target_region.shape[0]))
                    else:
                        cam_final = cam_resized
                    patch_attention_scores.append(score)

                    x, y = coord[0], coord[1]
                    x_scaled = int(x * scale_factor)    #经过缩放的左上角坐标
                    y_scaled = int(y * scale_factor)
                    patch_size_scaled = int(patch_size * scale_factor)
                    
                    # 确保坐标在范围内
                    x_end = min(x_scaled + patch_size_scaled, heatmap_w)
                    y_end = min(y_scaled + patch_size_scaled, heatmap_h)
                    x_scaled = max(0, x_scaled)
                    y_scaled = max(0, y_scaled)
                    
                    if x_end > x_scaled and y_end > y_scaled:
                        # 将注意力分数添加到对应区域
                        # wsi_heatmap[y_scaled:y_end, x_scaled:x_end] += score
                        wsi_heatmap[y_scaled:y_end, x_scaled:x_end] += cam_final
                        count_map[y_scaled:y_end, x_scaled:x_end] += 1
                
                # all_attention_scores.extend(patch_attention_scores)
                # all_coords.extend([coord for coord in patch_coords])
                '''
                需要注意，应该按照wsi来计算wsi_heatmap，每个batch都有多张wsis
                '''
                # 处理重叠区域，计算平均注意力
                count_map[count_map == 0] = 1  # 避免除零
                wsi_heatmap_avg = wsi_heatmap / count_map
        
                # 应用高斯模糊使热力图更平滑
                if blur_sigma > 0:
                    wsi_heatmap_avg = cv2.GaussianBlur(wsi_heatmap_avg, 
                                                    (2 * int(blur_sigma) + 1, 2 * int(blur_sigma) + 1), 
                                                    blur_sigma)
                # 归一化到[0, 1]
                if wsi_heatmap_avg.max() > 0:
                    wsi_heatmap_avg = (wsi_heatmap_avg - wsi_heatmap_avg.min()) / (wsi_heatmap_avg.max() - wsi_heatmap_avg.min() + 1e-8)
                
                wsi_heatmap_resized = cv2.resize(wsi_heatmap_avg, (thumbnail.shape[1], thumbnail.shape[0]))
        
                # 应用颜色映射
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * wsi_heatmap_resized), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                # 确保缩略图是RGB格式
                if len(thumbnail.shape) == 2:  # 灰度图
                    thumbnail = np.stack([thumbnail] * 3, axis = -1)
                
                # 叠加热力图
                superimposed_img = heatmap_colored * 0.5 + thumbnail * (1 - 0.5)
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

                fig, axes = plt.subplots(1, 3 if thumbnail is not None else 1, figsize = (15, 12))

                if thumbnail is not None:
                    axes[0].imshow(thumbnail)
                    axes[0].set_title('Thumbnail')
                    axes[0].axis('off')

                    im = axes[1].imshow(wsi_heatmap_avg, cmap = 'jet', alpha = 0.8)
                    axes[1].set_title('Heatmap')
                    axes[1].axis('off')

                    axes[2].imshow(superimposed_img)
                    axes[2].set_title('Superimposed')
                    axes[2].axis('off')

                    plt.colorbar(im, ax = axes[2], fraction = 0.046, pad = 0.04)

                else:
                    im = axes.imshow(wsi_heatmap_avg, cmap = 'jet', alpha = 0.8)
                    axes.set_title('Heatmap')
                    axes.axis('off')
                    plt.colorbar(im, ax = axes, fraction = 0.046, pad = 0.04)
                plt.tight_layout()

                save_path  = '/mnt/breast_cancer_multimodal/heatmap/' + guid + '.png'
                plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
                print('Saved:', save_path)
                plt.close()
                



