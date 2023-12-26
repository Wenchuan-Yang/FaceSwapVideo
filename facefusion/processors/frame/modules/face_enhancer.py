from typing import Any, List, Dict, Literal, Optional
from argparse import ArgumentParser
import cv2
import threading
import numpy
import onnxruntime

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.face_analyser import get_many_faces, clear_face_analyser
from facefusion.face_helper import warp_face, paste_back
from facefusion.content_analyser import clear_content_analyser
from facefusion.typing import Face, Frame, Update_Process, ProcessMode, ModelValue, OptionsWithModel
from facefusion.utilities import conditional_download, resolve_relative_path, is_image, is_video, is_file, is_download_done, create_metavar, update_status
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame import choices as frame_processors_choices

FRAME_PROCESSOR = None # frame processor
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()
NAME = 'FACEFUSION.FRAME_PROCESSOR.FACE_ENHANCER'
MODELS : Dict[str, ModelValue] =\
{
	'codeformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
		'path': resolve_relative_path('../.assets/models/codeformer.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gfpgan_1.2':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.2.onnx',
		'path': resolve_relative_path('../.assets/models/gfpgan_1.2.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gfpgan_1.3':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.3.onnx',
		'path': resolve_relative_path('../.assets/models/gfpgan_1.3.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gfpgan_1.4':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx',
		'path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gpen_bfr_256':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_256.onnx',
		'path': resolve_relative_path('../.assets/models/gpen_bfr_256.onnx'),
		'template': 'arcface_v2',
		'size': (128, 256)
	},
	'gpen_bfr_512':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx',
		'path': resolve_relative_path('../.assets/models/gpen_bfr_512.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'restoreformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/restoreformer.onnx',
		'path': resolve_relative_path('../.assets/models/restoreformer.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	}
}
OPTIONS : Optional[OptionsWithModel] = None


def get_frame_processor() -> Any:
	global FRAME_PROCESSOR

	with THREAD_LOCK:
		if FRAME_PROCESSOR is None:
			model_path = get_options('model').get('path')
			FRAME_PROCESSOR = onnxruntime.InferenceSession(model_path, providers = facefusion.globals.execution_providers)
	return FRAME_PROCESSOR


def clear_frame_processor() -> None:
	global FRAME_PROCESSOR

	FRAME_PROCESSOR = None


def get_options(key : Literal['model']) -> Any:
	global OPTIONS

	if OPTIONS is None:
		OPTIONS =\
		{
			'model': MODELS[frame_processors_globals.face_enhancer_model]
		}
	return OPTIONS.get(key)


def set_options(key : Literal['model'], value : Any) -> None:
	global OPTIONS

	OPTIONS[key] = value


def register_args(program : ArgumentParser) -> None:
	program.add_argument('--face-enhancer-model', help = wording.get('frame_processor_model_help'), dest = 'face_enhancer_model', default = 'gfpgan_1.4', choices = frame_processors_choices.face_enhancer_models)
	program.add_argument('--face-enhancer-blend', help = wording.get('frame_processor_blend_help'), dest = 'face_enhancer_blend', type = int, default = 80, choices = frame_processors_choices.face_enhancer_blend_range, metavar = create_metavar(frame_processors_choices.face_enhancer_blend_range))


def apply_args() -> None:
	frame_processors_globals.face_enhancer_model = 'gfpgan_1.4' # args.face_enhancer_model
	frame_processors_globals.face_enhancer_blend = 80 #args.face_enhancer_blend


def pre_check() -> bool:
	if not facefusion.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_url = get_options('model').get('url')
		conditional_download(download_directory_path, [ model_url ])
	return True


def pre_process(mode : ProcessMode) -> bool:
	model_url = get_options('model').get('url')
	model_path = get_options('model').get('path')
	if not facefusion.globals.skip_download and not is_download_done(model_url, model_path):
		update_status(wording.get('model_download_not_done') + wording.get('exclamation_mark'), NAME)
		return False
	elif not is_file(model_path):
		update_status(wording.get('model_file_not_present') + wording.get('exclamation_mark'), NAME)
		return False
	if mode in [ 'output', 'preview' ] and not is_image(facefusion.globals.target_path) and not is_video(facefusion.globals.target_path):
		update_status(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
		return False
	if mode == 'output' and not facefusion.globals.output_path:
		update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
		return False
	return True


def post_process() -> None:
	clear_frame_processor()
	clear_face_analyser()
	clear_content_analyser()
	read_static_image.cache_clear()


# 定义一个函数，用于增强目标人脸，参数为目标人脸和临时帧，返回值为帧
def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
	# 获取帧处理器
	frame_processor = get_frame_processor()
	# 获取模型模板
	model_template = get_options('model').get('template')
	# 获取模型大小
	model_size = get_options('model').get('size')
	# 对帧进行裁剪，并获取仿射矩阵
	crop_frame, affine_matrix = warp_face(temp_frame, target_face.kps, model_template, model_size)
	# 准备裁剪帧
	crop_frame = prepare_crop_frame(crop_frame)
	# 初始化帧处理器输入
	frame_processor_inputs = {}
	# 遍历帧处理器输入
	for frame_processor_input in frame_processor.get_inputs():
		# 如果输入名为input，则将裁剪帧赋值给输入
		if frame_processor_input.name == 'input':
			frame_processor_inputs[frame_processor_input.name] = crop_frame
		# 如果输入名为weight，则将1赋值给输入
		if frame_processor_input.name == 'weight':
			frame_processor_inputs[frame_processor_input.name] = numpy.array([ 1 ], dtype = numpy.double)
	# 获取线程信号量
	with THREAD_SEMAPHORE:
		# 运行帧处理器，获取处理后的帧
		crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
	# 归一化裁剪帧
	crop_frame = normalize_crop_frame(crop_frame)
	# 将裁剪帧粘贴到临时帧上，并设置模糊度
	paste_frame = paste_back(temp_frame, crop_frame, affine_matrix, facefusion.globals.face_mask_blur, (0, 0, 0, 0))
	# 将粘贴帧和临时帧混合
	temp_frame = blend_frame(temp_frame, paste_frame)
	# 返回混合后的帧
	return temp_frame


# 定义一个函数，用于准备裁剪帧
def prepare_crop_frame(crop_frame : Frame) -> Frame:
	# 将裁剪帧转换为RGB格式，并除以255
	crop_frame = crop_frame[:, :, ::-1] / 255.0
	# 将裁剪帧的像素值减去0.5，再除以0.5
	crop_frame = (crop_frame - 0.5) / 0.5
	# 将裁剪帧增加一个维度，并转换为float32类型
	crop_frame = numpy.expand_dims(crop_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	# 返回裁剪帧
	return crop_frame

# 定义一个函数normalize_crop_frame，用于将帧的值转换为[0, 255]的uint8类型
def normalize_crop_frame(crop_frame : Frame) -> Frame:
	# 将帧的值限制在[-1, 1]之间
	crop_frame = numpy.clip(crop_frame, -1, 1)
	# 将帧的值转换为[0, 1]
	crop_frame = (crop_frame + 1) / 2
	# 将帧的维度转换为[H, W, C]
	crop_frame = crop_frame.transpose(1, 2, 0)
	# 将帧的值转换为[0, 255]
	crop_frame = (crop_frame * 255.0).round()
	# 将帧的值转换为uint8类型
	crop_frame = crop_frame.astype(numpy.uint8)[:, :, ::-1]
	# 返回转换后的帧
	return crop_frame


# 定义一个函数blend_frame，用于将两个帧进行混合，返回混合后的帧
def blend_frame(temp_frame : Frame, paste_frame : Frame) -> Frame:
	# 计算face_enhancer_blend的值，用于混合两个帧
	face_enhancer_blend = 1 - (frame_processors_globals.face_enhancer_blend / 100)
	# 将两个帧进行混合，返回混合后的帧
	temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
	return temp_frame


# 定义一个函数process_frame，用于处理帧，返回处理后的帧
def process_frame(source_face : Face, reference_face : Face, temp_frame : Frame) -> Frame:
	# 获取帧中的多个脸部
	many_faces = get_many_faces(temp_frame)
	# 如果存在多个脸部，则遍历每一个脸部，对每一个脸部进行增强处理
	if many_faces:
		for target_face in many_faces:
			temp_frame = enhance_face(target_face, temp_frame)
	return temp_frame

# 
def process_frames(source_path : str, temp_frame_paths : List[str], update_progress : Update_Process) -> None:
	# 遍历临时帧路径列表
	for temp_frame_path in temp_frame_paths:
		# 读取临时帧
		temp_frame = read_image(temp_frame_path)
		# 处理帧
		result_frame = process_frame(None, None, temp_frame)
		# 写入处理后的帧
		write_image(temp_frame_path, result_frame)
		# 更新进度
		update_progress()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	# 读取目标帧
	target_frame = read_static_image(target_path)
	# 处理帧
	result_frame = process_frame(None, None, target_frame)
	# 写入处理后的帧
	write_image(output_path, result_frame)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	# 调用多进程帧处理函数
	frame_processors.multi_process_frames(None, temp_frame_paths, process_frames)
