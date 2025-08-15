import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import os
import sys
import threading

# 全局变量存储导入的模块
SAM_MODULES = None

def import_sam_modules():
    """延迟导入SAM模块 - 支持多种导入方式"""
    global SAM_MODULES
    if SAM_MODULES is not None:
        return SAM_MODULES
    
    print("尝试导入SAM模块...")
    
    # 获取当前文件的目录
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 在Jupyter notebook中__file__不存在，使用当前工作目录
        current_dir = os.getcwd()
        print(f"注意: 在Jupyter环境中，使用当前工作目录: {current_dir}")
    
    parent_dir = os.path.dirname(current_dir)
    
    # 方式1：尝试从segment_anything包导入（官方方式）
    try:
        print("方式1: 从segment_anything包导入（官方方式）...")
        from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
        SAM_MODULES = (sam_model_registry, SamPredictor, SamAutomaticMaskGenerator)
        print("✓ 方式1导入成功（官方segment_anything）")
        return SAM_MODULES
    except ImportError as e:
        print(f"✗ 方式1失败: {e}")
    
    # 方式2：尝试从segment_anything包导入（旧方式）
    try:
        print("方式2: 从segment_anything包导入（旧方式）...")
        from segment_anything.build_sam import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
        from segment_anything.predictor import SamPredictor
        from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
        SAM_MODULES = (build_sam_vit_b, build_sam_vit_l, build_sam_vit_h, SamPredictor, SamAutomaticMaskGenerator)
        print("✓ 方式2导入成功")
        return SAM_MODULES
    except ImportError as e:
        print(f"✗ 方式2失败: {e}")
    
    # 方式3：尝试从当前目录直接导入
    try:
        print("方式3: 从当前目录导入...")
        import importlib.util
        
        # 确保当前目录在路径中
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 导入build_sam
        build_sam_path = os.path.join(current_dir, "build_sam.py")
        if os.path.exists(build_sam_path):
            spec = importlib.util.spec_from_file_location("build_sam", build_sam_path)
            build_sam_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(build_sam_module)
            
            build_sam_vit_b = getattr(build_sam_module, 'build_sam_vit_b', None)
            build_sam_vit_l = getattr(build_sam_module, 'build_sam_vit_l', None)
            build_sam_vit_h = getattr(build_sam_module, 'build_sam_vit_h', None)
        else:
            raise ImportError("build_sam.py not found")
        
        # 导入predictor
        predictor_path = os.path.join(current_dir, "predictor.py")
        if os.path.exists(predictor_path):
            spec = importlib.util.spec_from_file_location("predictor", predictor_path)
            predictor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(predictor_module)
            
            SamPredictor = getattr(predictor_module, 'SamPredictor', None)
        else:
            raise ImportError("predictor.py not found")
        
        if all([build_sam_vit_b, build_sam_vit_l, build_sam_vit_h, SamPredictor]):
            sam_model_registry = getattr(build_sam_module, 'sam_model_registry', None)
            SAM_MODULES = (sam_model_registry, SamPredictor, None)
            print("✓ 方式3导入成功")
            return SAM_MODULES
        else:
            raise ImportError("Some functions not found")
            
    except Exception as e:
        print(f"✗ 方式3失败: {e}")
    
    # 方式4：尝试简单的直接导入
    try:
        print("方式4: 简单直接导入...")
        
        # 确保当前目录在路径中（重要！）
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"添加当前目录到路径: {current_dir}")
        
        from build_sam import sam_model_registry
        from predictor import SamPredictor
        SAM_MODULES = (sam_model_registry, SamPredictor, None)
        print("✓ 方式4导入成功")
        return SAM_MODULES
    except ImportError as e:
        print(f"✗ 方式4失败: {e}")
        print(f"当前目录: {current_dir}")
        print(f"sys.path中是否包含当前目录: {current_dir in sys.path}")
        
        # 检查文件是否存在
        build_sam_file = os.path.join(current_dir, "build_sam.py")
        predictor_file = os.path.join(current_dir, "predictor.py")
        print(f"build_sam.py存在: {os.path.exists(build_sam_file)}")
        print(f"predictor.py存在: {os.path.exists(predictor_file)}")
        
        if os.path.exists(build_sam_file):
            print(f"build_sam.py路径: {build_sam_file}")
        if os.path.exists(predictor_file):
            print(f"predictor.py路径: {predictor_file}")
    
    # 方式5：尝试从父目录导入
    try:
        print("方式5: 从父目录导入...")
        import importlib.util
        
        parent_build_sam = os.path.join(parent_dir, "build_sam.py")
        parent_predictor = os.path.join(parent_dir, "predictor.py")
        
        if os.path.exists(parent_build_sam) and os.path.exists(parent_predictor):
            # 导入build_sam
            spec = importlib.util.spec_from_file_location("build_sam", parent_build_sam)
            build_sam_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(build_sam_module)
            
            build_sam_vit_b = getattr(build_sam_module, 'build_sam_vit_b', None)
            build_sam_vit_l = getattr(build_sam_module, 'build_sam_vit_l', None)
            build_sam_vit_h = getattr(build_sam_module, 'build_sam_vit_h', None)
            
            # 导入predictor
            spec = importlib.util.spec_from_file_location("predictor", parent_predictor)
            predictor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(predictor_module)
            
            SamPredictor = getattr(predictor_module, 'SamPredictor', None)
            
            if all([build_sam_vit_b, build_sam_vit_l, build_sam_vit_h, SamPredictor]):
                sam_model_registry = getattr(build_sam_module, 'sam_model_registry', None)
                SAM_MODULES = (sam_model_registry, SamPredictor, None)
                print("✓ 方式5导入成功")
                return SAM_MODULES
        raise ImportError("Files not found in parent directory")        
    except Exception as e:
        print(f"✗ 方式5失败: {e}")
    
    # 方式6：使用模拟SAM模块（用于测试）
    try:
        print("方式6: 使用模拟SAM模块...")
        
        # 确保当前目录在路径中
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from mock_sam import sam_model_registry
        SAM_MODULES = (sam_model_registry, SamPredictor, None)
        print("✓ 方式6导入成功（使用模拟模块）")
        print("⚠️  注意：当前使用的是模拟SAM模块，仅用于界面测试")
        return SAM_MODULES
    except ImportError as e:
        print(f"✗ 方式6失败: {e}")
    
    print("✗ 所有导入方式都失败了")
    print("解决方案:")
    print("1. 运行 install_sam.py 安装 segment-anything")
    print("2. 或者使用 mock_sam.py 进行界面测试")
    print("3. 检查 build_sam.py 和 predictor.py 文件是否存在")
    
    SAM_MODULES = (None, None, None, None)
    return SAM_MODULES

# 添加支持中文路径的OpenCV函数
def cv_imread_unicode(file_path):
    """支持中文路径的cv2.imread"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def cv_imwrite_unicode(file_path, img):
    """支持中文路径的cv2.imwrite"""
    ext = os.path.splitext(file_path)[1]
    is_success, buffer = cv2.imencode(ext, img)
    if is_success:
        buffer.tofile(file_path)
        return True
    return False

# 简化版自动mask生成器类
class SimpleSamAutomaticMaskGenerator:
    """简化版的SAM自动mask生成器"""
    
    def __init__(self, model, points_per_side=16, pred_iou_thresh=0.6, stability_score_thresh=0.6, min_mask_region_area=100):
        from predictor import SamPredictor
        self.predictor = SamPredictor(model)
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

    def generate(self, image):
        """生成自动mask"""
        print(f"开始自动分割，参数: points_per_side={self.points_per_side}")
        
        # 设置图像
        self.predictor.set_image(image)
        
        h, w = image.shape[:2]
        print(f"图像尺寸: {w}x{h}")
        
        # 生成网格点
        y_coords = np.linspace(h * 0.1, h * 0.9, self.points_per_side)
        x_coords = np.linspace(w * 0.1, w * 0.9, self.points_per_side)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        print(f"生成了 {len(points)} 个采样点")
        
        masks = []
        batch_size = 32  # 减小批次大小以提高稳定性
        
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            batch_labels = np.ones(len(batch_points))
            
            try:
                print(f"处理批次 {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
                
                pred_masks, scores, _ = self.predictor.predict(
                    point_coords=batch_points,
                    point_labels=batch_labels,
                    multimask_output=True
                )
                
                # 处理每个预测结果
                for j in range(len(batch_points)):
                    if j < len(pred_masks):
                        # 选择最佳mask
                        if len(scores.shape) > 1:
                            mask_scores = scores[j]
                            best_idx = np.argmax(mask_scores)
                            best_mask = pred_masks[j][best_idx] if len(pred_masks[j].shape) > 2 else pred_masks[j]
                            best_score = mask_scores[best_idx]
                        else:
                            best_mask = pred_masks[j]
                            best_score = scores[j] if j < len(scores) else 0.5
                        
                        # 质量过滤
                        if best_score > self.pred_iou_thresh:
                            area = np.sum(best_mask)
                            
                            # 面积过滤
                            if area > self.min_mask_region_area:
                                bbox = self._mask_to_bbox(best_mask)
                                
                                mask_info = {
                                    'segmentation': best_mask,
                                    'area': int(area),
                                    'bbox': bbox,
                                    'predicted_iou': float(best_score),
                                    'point_coords': [batch_points[j].tolist()],
                                    'stability_score': float(best_score),
                                    'crop_box': [0, 0, w, h]
                                }
                                masks.append(mask_info)
                                
            except Exception as e:
                print(f"批次 {i//batch_size + 1} 处理失败: {e}")
                continue
        
        # 根据面积排序，保留较大的mask
        masks.sort(key=lambda x: x['area'], reverse=True)
        
        # 简单的NMS：移除重叠度过高的mask
        filtered_masks = []
        for mask in masks:
            if len(filtered_masks) >= 20:  # 限制最大数量
                break
                
            overlap = False
            for existing_mask in filtered_masks:
                if self._calculate_overlap(mask['segmentation'], existing_mask['segmentation']) > 0.5:
                    overlap = True
                    break
            
            if not overlap:
                filtered_masks.append(mask)
        
        print(f"自动分割完成: 生成了 {len(filtered_masks)} 个有效mask")
        return filtered_masks
    
    def _mask_to_bbox(self, mask):
        """将mask转换为bbox [x, y, w, h]"""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return [0, 0, 0, 0]
        
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
    
    def _calculate_overlap(self, mask1, mask2):
        """计算两个mask的重叠度"""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

class SAMGui:
    def __init__(self, root):
        self.root = root
        print("初始化SAMGui...")
        
        # 初始化变量
        self.original_image = None
        self.predictor = None
        self.mask_generator = None  # 添加自动mask生成器
        self.current_mask = None
        self.all_masks = None  # 存储自动生成的所有masks
        self.point_coords = []
        self.point_labels = []
        self.box_coords = None
        self.is_drawing_box = False
        self.box_start = None
        self.current_mode = "point"
        self.checkpoint_path = None
        self.model_folder = None  # 存储模型文件夹路径
        self.model_files = {"vit_b": None, "vit_l": None, "vit_h": None}  # 存储匹配的模型文件
        self.last_model_dir = None  # 记住上次选择模型文件的目录
        self.scale = 1.0
        self.img_x = 0
        self.img_y = 0
        self.sam_modules_loaded = True
        
        # 保存原始stdout并重定向print到日志
        self._orig_stdout = sys.stdout
        sys.stdout = self
        
        # 创建UI
        print("创建UI界面...")
        self.setup_ui()
        
        # 后台检查SAM模块
        print("启动模块检查...")
        self.check_sam_modules_async()
        
        print("SAMGui初始化完成")
    
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左右分割的PanedWindow
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧框架（原有的界面）
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=3)
        
        # 右侧框架（日志信息）
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)
        
        # 在左侧框架中创建原有控件
        self._create_main_controls(left_frame)
        
        # 在右侧框架中创建日志区域
        self._create_log_panel(right_frame)
    
    def _create_main_controls(self, parent):
        """创建主要控制界面"""
        # 控制面板
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # 模型配置区域
        model_frame = ttk.LabelFrame(control_frame, text="模型配置")
        model_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # 第一行：模型架构选择
        model_arch_frame = ttk.Frame(model_frame)
        model_arch_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        ttk.Label(model_arch_frame, text="模型架构:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_arch_var = tk.StringVar(value="vit_b")
        arch_frame = ttk.Frame(model_arch_frame)
        arch_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(arch_frame, text="ViT-B (最快)", variable=self.model_arch_var, 
                       value="vit_b").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(arch_frame, text="ViT-L (平衡)", variable=self.model_arch_var, 
                       value="vit_l").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(arch_frame, text="ViT-H (最佳)", variable=self.model_arch_var, 
                       value="vit_h").pack(side=tk.LEFT, padx=3)
        
        # 第二行：checkpoint文件选择
        checkpoint_frame = ttk.Frame(model_frame)
        checkpoint_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        ttk.Label(checkpoint_frame, text="模型文件:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.checkpoint_var = tk.StringVar(value="未选择模型文件")
        checkpoint_label = ttk.Label(checkpoint_frame, textvariable=self.checkpoint_var, 
                                   relief=tk.SUNKEN, width=35)
        checkpoint_label.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(checkpoint_frame, text="选择模型文件夹", 
                  command=self.select_model_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(checkpoint_frame, text="加载模型", 
                  command=self.load_model).pack(side=tk.LEFT, padx=(0, 10))
        
        # 模型状态指示
        self.model_status_var = tk.StringVar(value="检查模块中...")
        self.model_status_label = ttk.Label(checkpoint_frame, textvariable=self.model_status_var)
        self.model_status_label.pack(side=tk.LEFT, padx=(10, 0))
        self.update_status_color("orange")
        
        # 操作控制区域
        operation_frame = ttk.Frame(control_frame)
        operation_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 10))
        
        # 文件操作
        file_frame = ttk.LabelFrame(operation_frame, text="文件操作")
        file_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(file_frame, text="加载图像", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(file_frame, text="保存mask", command=self.save_mask).pack(side=tk.LEFT, padx=3)        
        ttk.Button(file_frame, text="保存叠加图", command=self.save_overlay).pack(side=tk.LEFT, padx=3)
        
        # 模式选择
        mode_frame = ttk.LabelFrame(operation_frame, text="操作模式")
        mode_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        self.mode_var = tk.StringVar(value="point")
        ttk.Radiobutton(mode_frame, text="点击模式", variable=self.mode_var, 
                       value="point", command=self.change_mode).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(mode_frame, text="框选模式", variable=self.mode_var, 
                       value="box", command=self.change_mode).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(mode_frame, text="自动分割", variable=self.mode_var, 
                       value="auto", command=self.change_mode).pack(side=tk.LEFT, padx=3)
        
        # 点击类型选择
        self.point_frame = ttk.LabelFrame(operation_frame, text="点击类型")
        self.point_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        self.point_type_var = tk.StringVar(value="foreground")
        ttk.Radiobutton(self.point_frame, text="前景点", variable=self.point_type_var, 
                       value="foreground").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(self.point_frame, text="背景点", variable=self.point_type_var, 
                       value="background").pack(side=tk.LEFT, padx=3)
        
        # 操作按钮
        action_frame = ttk.LabelFrame(operation_frame, text="操作")
        action_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(action_frame, text="清除", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="生成mask", command=self.generate_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="撤销", command=self.undo_last_point).pack(side=tk.LEFT, padx=2)
        
        # 图像显示区域
        image_frame = ttk.Frame(parent)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Canvas - 设置更小的初始尺寸
        self.canvas = tk.Canvas(image_frame, bg="white", width=400, height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # 状态栏
        self.status_var = tk.StringVar(value="正在初始化...")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def _create_log_panel(self, parent):
        """创建日志面板"""
        # 日志区域标题
        log_label_frame = ttk.LabelFrame(parent, text="日志信息")
        log_label_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建日志文本框和滚动条
        log_frame = ttk.Frame(log_label_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=20, width=50, wrap=tk.WORD,
                               font=("Consolas", 9), bg="#f8f8f8", fg="#333333")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 垂直滚动条
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # 日志控制按钮
        log_control_frame = ttk.Frame(log_label_frame)
        log_control_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_control_frame, text="保存日志", command=self.save_log).pack(side=tk.LEFT, padx=2)
        
        # 自动滚动复选框
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_control_frame, text="自动滚动", 
                       variable=self.auto_scroll_var).pack(side=tk.RIGHT, padx=2)

    def write(self, msg):
        """print 重定向：同时写入原stdout和日志窗"""
        self._orig_stdout.write(msg)
        if hasattr(self, 'log_text'):
            try:
                # 检查widget是否仍然存在
                if self.log_text.winfo_exists():
                    self.log_text.insert("end", msg)
                    if hasattr(self, 'auto_scroll_var') and self.auto_scroll_var.get():
                        self.log_text.see("end")
            except tk.TclError:
                # Widget已经被销毁，只输出到原始stdout
                pass

    def flush(self):
        """支持flush调用"""
        self._orig_stdout.flush()

    def clear_log(self):
        """清空日志"""
        if hasattr(self, 'log_text'):
            try:
                if self.log_text.winfo_exists():
                    self.log_text.delete(1.0, tk.END)
            except tk.TclError:
                pass

    def save_log(self):
        """保存日志到文件"""
        if hasattr(self, 'log_text'):
            try:
                if not self.log_text.winfo_exists():
                    messagebox.showerror("错误", "日志组件已不可用")
                    return
                    
                file_path = filedialog.asksaveasfilename(
                    title="保存日志",
                    defaultextension=".txt",
                    filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
                )
                
                if file_path:
                    log_content = self.log_text.get(1.0, tk.END)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(log_content)
                    messagebox.showinfo("成功", f"日志已保存到: {file_path}")
            except tk.TclError:
                messagebox.showerror("错误", "日志组件已不可用")
            except Exception as e:
                messagebox.showerror("错误", f"保存日志失败: {str(e)}")

    def __del__(self):
        """析构函数，恢复原始stdout"""
        try:
            if hasattr(self, '_orig_stdout'):
                sys.stdout = self._orig_stdout
        except:
            pass

    def check_sam_modules_async(self):
        """异步检查SAM模块是否可用"""
        def check_modules():
            try:
                modules = import_sam_modules()
                if modules[0] is None:
                    self.root.after(0, lambda: self.show_module_error())
                else:
                    self.sam_modules_loaded = True
                    self.root.after(0, lambda: self.update_module_status(True))
            except Exception as e:
                self.root.after(0, lambda: self.show_module_error(str(e)))
        
        # 在后台线程中检查模块
        thread = threading.Thread(target=check_modules, daemon=True)
        thread.start()
    
    def show_module_error(self, error_msg=""):
        """显示模块加载错误"""
        self.model_status_var.set("模块加载失败")
        self.update_status_color("red")
        self.status_var.set("SAM模块不可用，请检查安装")
        print(f"SAM模块加载失败: {error_msg}")
    
    def update_module_status(self, success):
        """更新模块状态"""
        if success:
            self.model_status_var.set("模块已加载")
            self.update_status_color("orange")
            self.status_var.set("SAM模块已加载，请选择模型文件")
        else:
            self.show_module_error()
    
    def update_status_color(self, color):
        """更新状态标签颜色"""
        try:
            style = ttk.Style()
            if color == "red":
                style.configure("Red.TLabel", foreground="red")
                self.model_status_label.configure(style="Red.TLabel")
            elif color == "green":
                style.configure("Green.TLabel", foreground="green")
                self.model_status_label.configure(style="Green.TLabel")
            elif color == "orange":
                style.configure("Orange.TLabel", foreground="orange")
                self.model_status_label.configure(style="Orange.TLabel")
        except:
            pass
    
    def select_model_folder(self):
        """选择包含SAM模型文件的文件夹"""
        if not self.sam_modules_loaded:
            messagebox.showwarning("警告", "SAM模块尚未加载完成，请稍候")
            return
        
        # 优先使用上次选择的目录，否则使用用户主目录
        if self.last_model_dir and os.path.exists(self.last_model_dir):
            initial_dir = self.last_model_dir
        else:
            initial_dir = os.path.expanduser("~")
        
        folder_path = filedialog.askdirectory(
            title="选择包含SAM模型文件的文件夹",
            initialdir=initial_dir
        )
        
        if folder_path:
            try:
                # 记住这个目录
                self.last_model_dir = folder_path
                self.model_folder = folder_path
                
                # 扫描文件夹中的.pth文件
                self.scan_model_files(folder_path)
                
            except Exception as e:
                messagebox.showerror("错误", f"文件夹扫描失败: {str(e)}")

    def scan_model_files(self, folder_path):
        """扫描文件夹中的模型文件并自动匹配"""
        # 重置模型文件字典
        self.model_files = {"vit_b": None, "vit_l": None, "vit_h": None}
        
        # 获取文件夹中所有.pth文件
        pth_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith('.pth'):
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path):
                    pth_files.append(file)
        
        if not pth_files:
            messagebox.showwarning("警告", "所选文件夹中没有找到.pth文件")
            self.checkpoint_var.set("未找到模型文件")
            self.model_status_var.set("未找到文件")
            self.update_status_color("red")
            return
        
        # 自动匹配模型文件
        matched_files = {}
        patterns = {
            "vit_b": ["vit_b", "sam_vit_b", "base", "_b_", "b_01ec64", "01ec64"],
            "vit_l": ["vit_l", "sam_vit_l", "large", "_l_", "l_0b3195", "0b3195"], 
            "vit_h": ["vit_h", "sam_vit_h", "huge", "_h_", "h_4b8939", "4b8939"]
        }
        
        print(f"找到的.pth文件: {pth_files}")
        
        for file in pth_files:
            file_lower = file.lower()
            print(f"检查文件: {file}")
            for arch, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in file_lower and self.model_files[arch] is None:
                        self.model_files[arch] = os.path.join(folder_path, file)
                        matched_files[arch] = file
                        print(f"  匹配 {arch.upper()}: {keyword} -> {file}")
                        break                
                    if self.model_files[arch] is not None:
                        break
        
        # 更新UI显示
        if matched_files:
            match_info = []
            for arch, file in matched_files.items():
                if file:
                    match_info.append(f"{arch.upper()}: {file}")
            
            self.checkpoint_var.set(f"已匹配 {len(matched_files)} 个模型")
            self.model_status_var.set("已扫描，未加载")
            self.update_status_color("orange")
            self.status_var.set(f"文件夹: {os.path.basename(folder_path)} | " + " | ".join(match_info))
        else:
            self.checkpoint_var.set("无法自动匹配")
            self.model_status_var.set("匹配失败")
            self.update_status_color("red")
            self.status_var.set(f"找到 {len(pth_files)} 个.pth文件，但无法自动匹配模型架构")

    def load_model(self):
        """加载SAM模型 - 优化版本"""
        if not self.sam_modules_loaded:
            messagebox.showwarning("警告", "SAM模块尚未加载完成")
            return
        
        # 检查是否已选择模型文件夹并匹配到模型
        arch = self.model_arch_var.get()
        if not self.model_files.get(arch):
            messagebox.showwarning("警告", f"请先选择模型文件夹，或者文件夹中没有找到 {arch.upper()} 模型")
            return
        self.checkpoint_path = self.model_files[arch]
        
        if not os.path.exists(self.checkpoint_path):
            messagebox.showerror("错误", f"{arch.upper()} 模型文件不存在")
            return
        
        # 在后台线程中加载模型
        def load_model_thread():
            try:
                print(f"正在加载 {arch} 模型...")
                print(f"模型文件路径: {self.checkpoint_path}")
                
                # 获取SAM模块
                sam_modules = import_sam_modules()
                sam_model_registry, SamPredictor, SamAutomaticMaskGenerator = sam_modules[:3]
                
                if sam_model_registry is None:
                    raise ImportError("SAM模块导入失败")
                
                # 构建模型
                if isinstance(sam_model_registry, dict):
                    model_type = f"vit_{arch[-1]}"
                    sam_model = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
                else:
                    # 如果是函数形式
                    sam_model = sam_model_registry(checkpoint=str(self.checkpoint_path))
                
                print("SAM模型构建完成")
                
                # 创建预测器
                predictor = SamPredictor(sam_model)
                print("预测器创建完成")
                
                # 创建自动mask生成器
                mask_generator = None
                try:
                    if SamAutomaticMaskGenerator is not None:
                        # 尝试使用官方版本
                        mask_generator = SamAutomaticMaskGenerator(sam_model)
                        print("✓ 使用官方SamAutomaticMaskGenerator")
                    else:
                        raise ImportError("官方版本不可用")
                        
                except Exception as e:
                    print(f"官方自动生成器创建失败: {e}")
                    try:
                        # 使用简化版本
                        mask_generator = SimpleSamAutomaticMaskGenerator(sam_model)
                        print("✓ 使用简化版SimpleSamAutomaticMaskGenerator")
                    except Exception as e2:
                        print(f"简化版自动生成器也失败: {e2}")
                        mask_generator = None
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self.on_model_loaded(predictor, arch, mask_generator))
                
            except Exception as e:
                error_msg = str(e)
                print(f"模型加载失败: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.on_model_load_failed(error_msg))
        
        # 更新UI状态
        self.status_var.set(f"正在加载 {arch.upper()} 模型...")
        self.model_status_var.set("加载中...")
        self.update_status_color("orange")
        self.root.update()
        
        # 启动加载线程
        thread = threading.Thread(target=load_model_thread, daemon=True)
        thread.start()

    def on_model_loaded(self, predictor, arch, mask_generator=None):
        """模型加载完成回调"""
        self.predictor = predictor
        self.mask_generator = mask_generator
        print(f"设置预测器: {type(predictor)}")
        
        if mask_generator is not None:
            print(f"设置自动mask生成器: {type(mask_generator)}")
        else:
            print("⚠️  自动mask生成器不可用，仅支持交互式分割")
        
        # 如果已经加载了图像，重新设置
        if self.original_image is not None:
            print("检测到已有图像，重新设置到新的预测器...")
            self.predictor.set_image(self.original_image)
            print("图像已重新设置到SAM预测器")
        
        self.model_status_var.set("已加载")
        self.update_status_color("green")
        if self.mask_generator is not None:
            self.status_var.set(f"模型加载完成: {arch.upper()} (支持自动分割)")
        else:
            self.status_var.set(f"模型加载完成: {arch.upper()} (仅支持交互式分割)")
        print(f"模型加载成功! 预测器已准备就绪: {self.predictor is not None}")
        print(f"自动分割功能: {'可用' if self.mask_generator is not None else '不可用'}")

    def on_model_load_failed(self, error_msg):
        """模型加载失败回调"""
        messagebox.showerror("错误", f"模型加载失败: {error_msg}")
        self.model_status_var.set("加载失败")
        self.update_status_color("red")
        self.status_var.set("模型加载失败")
        self.predictor = None

    def generate_mask(self):
        """生成分割mask - 优化版本"""
        if self.predictor is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        try:
            if self.current_mode == "auto":
                # 自动分割模式
                if self.mask_generator is None:
                    messagebox.showwarning("警告", "当前模型不支持自动分割功能\n请使用点击模式或框选模式")
                    return
                
                print("=" * 50)
                print("开始自动分割...")
                self.status_var.set("正在进行自动分割...")
                self.root.update()
                
                # 在后台线程中执行自动分割
                def auto_segment():
                    try:
                        masks = self.mask_generator.generate(self.original_image)
                        self.root.after(0, lambda: self.on_auto_segment_complete(masks))
                    except Exception as e:
                        error_msg = str(e)
                        self.root.after(0, lambda: self.on_auto_segment_failed(error_msg))
                
                thread = threading.Thread(target=auto_segment, daemon=True)
                thread.start()
                return
            
            # 交互式分割模式（点击或框选）
            point_coords = np.array(self.point_coords) if self.point_coords else None
            point_labels = np.array(self.point_labels) if self.point_labels else None
            box = np.array(self.box_coords) if self.box_coords else None
            
            if point_coords is None and box is None:
                messagebox.showwarning("警告", "请先添加点或框选区域")
                return
            
            self.status_var.set("正在生成mask...")
            self.root.update()
            
            print(f"交互式分割参数: 点={len(self.point_coords) if self.point_coords else 0}, 框={'有' if box is not None else '无'}")
            
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
            
            # 选择最佳mask
            if len(scores) > 1:
                best_mask_idx = np.argmax(scores)
                self.current_mask = masks[best_mask_idx]
                print(f"选择最佳mask (索引 {best_mask_idx})，质量分数: {scores[best_mask_idx]:.3f}")
            else:
                self.current_mask = masks[0]
                print(f"使用单个mask，质量分数: {scores[0]:.3f}")
            
            self.display_image_on_canvas()
            best_score = scores[np.argmax(scores)] if len(scores) > 1 else scores[0]
            self.status_var.set(f"Mask生成完成，质量分数: {best_score:.3f}")
            
        except Exception as e:
            print(f"Mask生成失败: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"Mask生成失败: {str(e)}")
            self.status_var.set("Mask生成失败")

    def on_auto_segment_complete(self, masks):
        """自动分割完成回调"""
        print(f"自动分割完成，生成了 {len(masks)} 个mask")
        
        if len(masks) > 0:
            # 选择面积最大的mask
            best_mask = None
            best_area = 0
            best_mask_info = None
            
            for mask_data in masks:
                mask = mask_data['segmentation']
                area = mask_data.get('area', np.sum(mask))
                
                if area > best_area:
                    best_area = area
                    best_mask = mask
                    best_mask_info = mask_data
            
            if best_mask is not None:
                self.current_mask = best_mask
                self.all_masks = masks
                self.display_image_on_canvas()
                
                # 显示详细信息
                iou_score = best_mask_info.get('predicted_iou', 0)
                self.status_var.set(f"自动分割完成，生成了 {len(masks)} 个mask，显示最大的一个 (面积: {best_area}, IoU: {iou_score:.3f})")
                print(f"选择最大mask，面积: {best_area} 像素, IoU: {iou_score:.3f}")
                print("=" * 50)
            else:
                self.status_var.set("自动分割未找到有效mask")
                print("=" * 50)
        else:
            self.status_var.set("自动分割未生成任何mask")
            print("=" * 50)

    def on_auto_segment_failed(self, error_msg):
        """自动分割失败回调"""
        print("=" * 50)
        print(f"自动分割失败: {error_msg}")
        print("=" * 50)
        messagebox.showerror("错误", f"自动分割失败: {error_msg}")
        self.status_var.set("自动分割失败")

    def load_image(self):
        """加载图像文件"""
        if self.predictor is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                print(f"正在加载图像: {file_path}")
                # 读取图像
                self.original_image = cv_imread_unicode(file_path)
                if self.original_image is None:
                    raise ValueError("无法读取图像文件")
                
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                print(f"图像读取成功，形状: {self.original_image.shape}")
                
                # 设置图像到predictor
                self.predictor.set_image(self.original_image)
                
                # 显示图像
                self.display_image_on_canvas()
                
                # 重置状态
                self.clear_all()
                
                filename = os.path.basename(file_path)
                self.status_var.set(f"图像加载完成: {filename}")
                
            except Exception as e:
                messagebox.showerror("错误", f"图像加载失败: {str(e)}")

    def display_image_on_canvas(self):
        """显示图像到画布"""
        if self.original_image is None:
            return
        
        # 获取Canvas尺寸
        self.root.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:
            self.root.after(50, self.display_image_on_canvas)
            return
        
        # 计算缩放比例
        img_height, img_width = self.original_image.shape[:2]
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale = min(scale_x, scale_y) * 0.95
        
        # 缩放图像
        new_width = int(img_width * self.scale)
        new_height = int(img_height * self.scale)
        
        display_img = cv2.resize(self.original_image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
        
        # 如果有mask，叠加显示
        if self.current_mask is not None:
            mask_resized = cv2.resize(self.current_mask.astype(np.uint8), (new_width, new_height))
            colored_mask = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            colored_mask[mask_resized > 0] = [0, 255, 0]
            display_img = cv2.addWeighted(display_img, 0.7, colored_mask, 0.3, 0)
        
        # 绘制点和框
        self.draw_annotations(display_img)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 更新Canvas
        self.canvas.delete("all")
        self.img_x = (canvas_width - new_width) // 2
        self.img_y = (canvas_height - new_height) // 2
        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.photo)

    def draw_annotations(self, image):
        """在图像上绘制标注点和框"""
        # 绘制点
        for coord, label in zip(self.point_coords, self.point_labels):
            x, y = int(coord[0] * self.scale), int(coord[1] * self.scale)
            color = (255, 0, 0) if label == 1 else (0, 0, 255)
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 2)
        
        # 绘制框
        if self.box_coords is not None:
            x1, y1, x2, y2 = self.box_coords
            x1, y1 = int(x1 * self.scale), int(y1 * self.scale)
            x2, y2 = int(x2 * self.scale), int(y2 * self.scale)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """将Canvas坐标转换为图像坐标"""
        if self.original_image is None:
            return None, None
        
        img_x = (canvas_x - self.img_x) / self.scale
        img_y = (canvas_y - self.img_y) / self.scale
        
        # 确保坐标在图像范围内
        img_height, img_width = self.original_image.shape[:2]
        img_x = max(0, min(img_width - 1, img_x))
        img_y = max(0, min(img_height - 1, img_y))
        
        return img_x, img_y

    def on_canvas_click(self, event):
        """处理Canvas点击事件"""
        if self.original_image is None:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        if img_x is None:
            return
        
        if self.current_mode == "point":
            self.point_coords.append([img_x, img_y])
            label = 1 if self.point_type_var.get() == "foreground" else 0
            self.point_labels.append(label)
            self.display_image_on_canvas()
        elif self.current_mode == "box":
            self.is_drawing_box = True
            self.box_start = (img_x, img_y)

    def on_canvas_drag(self, event):
        """处理Canvas拖拽事件"""
        if self.current_mode == "box" and self.is_drawing_box and self.box_start:
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            if img_x is not None and img_y is not None:
                x1, y1 = self.box_start
                self.box_coords = [min(x1, img_x), min(y1, img_y), 
                                 max(x1, img_x), max(y1, img_y)]
                self.display_image_on_canvas()

    def on_canvas_release(self, event):
        """处理Canvas释放事件"""
        if self.current_mode == "box" and self.is_drawing_box:
            self.is_drawing_box = False
            if self.box_coords:
                self.generate_mask()

    def change_mode(self):
        """切换操作模式"""
        self.current_mode = self.mode_var.get()
        if self.current_mode == "point":
            self.point_frame.pack(side=tk.LEFT, padx=(0, 10))
        else:
            self.point_frame.pack_forget()

    def undo_last_point(self):
        """撤销最后一个点"""
        if self.point_coords:
            self.point_coords.pop()
            self.point_labels.pop()
            self.display_image_on_canvas()
            self.status_var.set("已撤销最后一个点")
        else:
            messagebox.showinfo("提示", "没有可撤销的点")

    def clear_all(self):
        """清除所有标注"""
        self.point_coords = []
        self.point_labels = []
        self.box_coords = None
        self.current_mask = None
        self.is_drawing_box = False
        self.box_start = None
        
        if self.original_image is not None:
            self.display_image_on_canvas()
        
        self.status_var.set("已清除所有标注")

    def save_mask(self):
        """保存当前mask"""
        if self.current_mask is None:
            messagebox.showwarning("警告", "没有可保存的mask")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存mask",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                mask_img = (self.current_mask > 0).astype(np.uint8) * 255
                cv_imwrite_unicode(file_path, mask_img)
                self.status_var.set(f"Mask已保存: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"保存mask失败: {str(e)}")

    def save_overlay(self):
        """保存带mask叠加的图像"""
        if self.current_mask is None or self.original_image is None:
            messagebox.showwarning("警告", "没有可保存的叠加图像")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存叠加图像",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                overlay_img = self.original_image.copy()
                colored_mask = np.zeros_like(overlay_img)
                colored_mask[self.current_mask > 0] = [0, 255, 0]
                result = cv2.addWeighted(overlay_img, 0.7, colored_mask, 0.3, 0)
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv_imwrite_unicode(file_path, result_bgr)
                self.status_var.set(f"叠加图像已保存: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

# 主程序入口
def main():
    try:
        print("启动SAM GUI应用程序...")
        
        # 创建主窗口
        root = tk.Tk()
        root.title("SAM 图像分割工具")
        root.geometry("1200x800")
        root.minsize(800, 600)
        
        # 添加关闭事件处理
        def on_closing():
            try:
                # 恢复原始stdout
                if hasattr(app, '_orig_stdout'):
                    sys.stdout = app._orig_stdout
                root.destroy()
            except:
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 创建应用程序
        app = SAMGui(root)
        
        print("GUI应用程序已启动")
        print("窗口标题: SAM 图像分割工具")
        print("窗口大小: 1200x800")
        
        # 启动主循环
        root.mainloop()
        
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()