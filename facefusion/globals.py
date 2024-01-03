from typing import List, Optional

from facefusion.typing import FaceSelectorMode, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, OutputVideoEncoder, FaceDetectorModel, FaceRecognizerModel, TempFrameFormat, Padding

# general
source_path : Optional[str] = '.assets/examples/source.jpg'
target_path : Optional[str] = '.assets/examples/target-1080p.mp4'
output_path : Optional[str] = '.assets/examples'
# misc
skip_download : Optional[bool] = False # 跳过下载#####################
headless : Optional[bool] = False # run the program in headless mode
# execution
execution_providers : List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # 计算资源########################
execution_thread_count : Optional[int] = 10 # 线程数
execution_queue_count : Optional[int] = 1 # 队列数
max_memory : Optional[int] = None # 限制内存
# face analyser
face_analyser_order : Optional[FaceAnalyserOrder] = 'left-right'
face_analyser_age : Optional[FaceAnalyserAge] = None
face_analyser_gender : Optional[FaceAnalyserGender] = None
face_detector_model : Optional[FaceDetectorModel] = 'retinaface' # 人脸检测模型
face_detector_size : Optional[str] = '640x640' # 人脸检测尺寸
face_detector_score : Optional[float] = 0.5 # 人脸检测分数
face_recognizer_model : Optional[FaceRecognizerModel] = 'arcface_inswapper' # 人脸识别模型
# face selector
face_selector_mode : Optional[FaceSelectorMode] = 'reference' 
reference_face_position : Optional[int] = 0 # 参考人脸的位置
reference_face_distance : Optional[float] = 0.6 # 指定参考面和目标面之间的距离(相似度)
reference_frame_number : Optional[int] = 0 # 指定参考帧的编号
# face mask
face_mask_blur : Optional[float] = 0.3 # mask的模糊量
face_mask_padding : Optional[Padding] = (0, 0, 0, 0) # 以百分比指定mask填充（上、右、下、左）
# frame extraction
trim_frame_start : Optional[int] = None # 开始执行操作的帧
trim_frame_end : Optional[int] = 10 #None # 结束执行操作的帧 ################## 测试值：10
temp_frame_format : Optional[TempFrameFormat] = 'jpg' # 中间格式
temp_frame_quality : Optional[int] = 90 # 用于帧提取的图像质量
keep_temp : Optional[bool] = False # None # 处理后保留临时帧
# output creation
output_image_quality : Optional[int] = 90 # 输出图像的质量
output_video_encoder : Optional[OutputVideoEncoder] = 'libx264' # 输出视频的编码器
output_video_quality : Optional[int] = 90 # 输出视频的质量
keep_fps : Optional[bool] = True # 保留目标的每秒帧数 (fps)
skip_audio : Optional[bool] = False # 忽略目标中的音频
# frame processors
frame_processors : List[str] = ['face_swapper', 'face_enhancer'] # 处理器
# uis
ui_layouts : List[str] = []

face_debugger_items = 'face_debugger_items'
face_enhancer_model = 'gfpgan_1.4'
face_enhancer_blend = 80
face_swapper_model = 'inswapper_128'
face_recognizer_model = 'arcface_inswapper'
frame_enhancer_model = 'real_esrgan_x2plus'
frame_enhancer_blend = 80
temp_path = "temp"