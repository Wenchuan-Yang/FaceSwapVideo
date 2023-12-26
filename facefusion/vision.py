from typing import Optional
from functools import lru_cache
import cv2

from facefusion.typing import Frame


# 定义一个函数，用于获取视频帧，参数为视频路径和帧号，返回值为帧图像
def get_video_frame(video_path : str, frame_number : int = 0) -> Optional[Frame]:
	# 如果视频路径存在
	if video_path:
		# 创建视频捕获对象
		video_capture = cv2.VideoCapture(video_path)
		# 如果视频捕获对象已打开
		if video_capture.isOpened():
			# 获取视频总帧数
			frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) # cap prop frame count
			# 设置视频捕获位置为帧号
			video_capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1)) # cap prop pos frames
			# 读取帧
			has_frame, frame = video_capture.read()
			# 释放视频捕获对象
			video_capture.release()
			# 如果读取到帧
			if has_frame:
				# 返回帧图像
				return frame
	# 如果没有读取到帧，返回None
	return None


# 定义一个函数，用于检测视频帧率
def detect_fps(video_path : str) -> Optional[float]:
	# 如果视频路径存在
	if video_path:
		# 创建视频捕获对象
		video_capture = cv2.VideoCapture(video_path)
		# 如果视频捕获对象打开
		if video_capture.isOpened():
			# 返回视频捕获对象的帧率
			return video_capture.get(cv2.CAP_PROP_FPS)
	# 返回None
	return None

# 定义一个函数，用于计算视频总帧数
def count_video_frame_total(video_path : str) -> int:
	# 如果视频路径存在
	if video_path:
		# 创建一个视频捕获对象
		video_capture = cv2.VideoCapture(video_path)
		# 如果视频捕获对象被打开
		if video_capture.isOpened():
			# 获取视频总帧数
			video_frame_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
			# 释放视频捕获对象
			video_capture.release()
			# 返回视频总帧数
			return video_frame_total
	# 如果视频路径不存在，返回0
	return 0


# 定义一个函数，用于将帧颜色规范化
def normalize_frame_color(frame : Frame) -> Frame:
	# 将帧从BGR转换为RGB
	return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# 定义一个函数，用于调整帧的尺寸，参数为帧，最大宽度，最大高度，返回值为调整后的帧
def resize_frame_dimension(frame : Frame, max_width : int, max_height : int) -> Frame:
	# 获取帧的高度和宽度
	height, width = frame.shape[:2]
	# 如果帧的高度大于最大高度或者帧的宽度大于最大宽度
	if height > max_height or width > max_width:
		# 计算缩放比例
		scale = min(max_height / height, max_width / width)
		# 计算新的宽度
		new_width = int(width * scale)
		# 计算新的高度
		new_height = int(height * scale)
		# 返回调整后的帧
		return cv2.resize(frame, (new_width, new_height))
	# 否则返回原帧
	return frame


# 定义一个函数，用于读取静态图像，参数为图像路径，返回值为读取的图像
@lru_cache(maxsize = 128)
def read_static_image(image_path : str) -> Optional[Frame]:
	# 返回读取的图像
	return read_image(image_path)


# 定义一个函数，读取图片，参数为图片路径，返回值为Frame
def read_image(image_path : str) -> Optional[Frame]:
	# 如果图片路径存在
	if image_path:
		# 返回读取的图片
		return cv2.imread(image_path)
	# 否则返回None
	return None


# 定义一个函数，写入图片，参数为图片路径和Frame，返回值为布尔值
def write_image(image_path : str, frame : Frame) -> bool:
	# 如果图片路径存在
	if image_path:
		# 返回写入的图片
		return cv2.imwrite(image_path, frame)
	# 否则返回False
	return False
