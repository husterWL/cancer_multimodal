import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

class GradCAM():
    def __init__(self, model, target_layer = None, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """
        Args:
            model: 模型
            target_layer: 目标卷积层（通常是最后一个卷积层）
        """
        self.model = model
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
        
        return cam.squeeze().cpu().numpy(), class_idx

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
    def batch_gradcam_visualization(self, dataloader, num_images=5):
        """批量生成Grad-CAM可视化"""
        
        
        images_processed = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if images_processed >= num_images:
                    break
                    
                for i in range(len(images)):
                    if images_processed >= num_images:
                        break
                        
                    image_tensor = images[i:i+1].to(self.device)
                    original_image = images[i].permute(1, 2, 0).cpu().numpy()
                    
                    # 反归一化
                    original_image = (original_image * np.array([0.229, 0.224, 0.226]) + 
                                np.array([0.485, 0.456, 0.406]))
                    original_image = np.clip(original_image, 0, 1)
                    
                    # 生成Grad-CAM
                    try:
                        heatmap, pred_class = self.create_gradcam_for_model(
                            image_tensor, original_image
                        )
                        images_processed += 1
                        
                        # 可以保存结果
                        plt.savefig(f'gradcam_result_{images_processed}.png', 
                                bbox_inches='tight', dpi=300)
                        plt.close()
                        
                    except Exception as e:
                        print(f"处理图像时出错: {e}")
                        continue

# 使用示例
# if __name__ == "__main__":
#     # 假设你已经加载了模型和数据
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Univision(config).to(device)
#     model.load_state_dict(torch.load("your_model_path.pth"))
    
#     # 测试单个图像
#     test_gradcam_on_single_image(model, test_dataloader, device)
    
#     # 或者批量处理
#     batch_gradcam_visualization(model, test_dataloader, device, num_images=10)
