from typing import Any, List, Dict, Literal, Optional
from argparse import ArgumentParser
import threading
import numpy
import onnx
import onnxruntime
from onnx import numpy_helper

import os
import sys

# 获取当前文件的绝对路径和文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append("/mnt/c/Python/facefusion")

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.face_analyser import get_one_face, get_many_faces, find_similar_faces, clear_face_analyser
from facefusion.face_helper import warp_face, paste_back
from facefusion.face_reference import get_face_reference
from facefusion.content_analyser import clear_content_analyser
from facefusion.typing import Face, Frame, Update_Process, ProcessMode, ModelValue, OptionsWithModel, Embedding
from facefusion.utilities import conditional_download, resolve_relative_path, is_image, is_video, is_file, is_download_done, update_status
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame import choices as frame_processors_choices

FRAME_PROCESSOR = None # frame processor
MODEL_MATRIX = None # model matrix
THREAD_LOCK : threading.Lock = threading.Lock()
NAME = 'FACEFUSION.FRAME_PROCESSOR.FACE_SWAPPER' # facefusion.frame processor
MODELS : Dict[str, ModelValue] =\
{
	'blendswap_256':
	{
		'type': 'blendswap',
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/blendswap_256.onnx',
		'path': resolve_relative_path('../.assets/models/blendswap_256.onnx'),
		'template': 'ffhq',
		'size': (512, 256),
		'mean': [ 0.0, 0.0, 0.0 ],
		'standard_deviation': [ 1.0, 1.0, 1.0 ]
	},
	'inswapper_128':
	{
		'type': 'inswapper',
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
		'path': resolve_relative_path('../.assets/models/inswapper_128.onnx'),
		'template': 'arcface_v2',
		'size': (128, 128),
		'mean': [ 0.0, 0.0, 0.0 ],
		'standard_deviation': [ 1.0, 1.0, 1.0 ]
	},
	'inswapper_128_fp16':
	{
		'type': 'inswapper',
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
		'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx'),
		'template': 'arcface_v2',
		'size': (128, 128),
		'mean': [ 0.0, 0.0, 0.0 ],
		'standard_deviation': [ 1.0, 1.0, 1.0 ]
	},
	'simswap_256':
	{
		'type': 'simswap',
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_256.onnx',
		'path': resolve_relative_path('../.assets/models/simswap_256.onnx'),
		'template': 'arcface_v1',
		'size': (112, 256),
		'mean': [ 0.485, 0.456, 0.406 ],
		'standard_deviation': [ 0.229, 0.224, 0.225 ]
	},
	'simswap_512_unofficial':
	{
		'type': 'simswap',
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_512_unofficial.onnx',
		'path': resolve_relative_path('../.assets/models/simswap_512_unofficial.onnx'),
		'template': 'arcface_v1',
		'size': (112, 512),
		'mean': [ 0.0, 0.0, 0.0 ],
		'standard_deviation': [ 1.0, 1.0, 1.0 ]
	}
}
OPTIONS : Optional[OptionsWithModel] = None


# 定义一个函数，返回类型为Any，用于获取帧处理器
def get_frame_processor() -> Any:
	# 声明一个全局变量FRAME_PROCESSOR
	global FRAME_PROCESSOR

	# 使用线程锁，确保线程安全
	with THREAD_LOCK:
		# 如果FRAME_PROCESSOR为None，则获取模型路径
		if FRAME_PROCESSOR is None:
			model_path = get_options('model').get('path')
			# 使用onnxruntime初始化FRAME_PROCESSOR，并设置执行提供者
			FRAME_PROCESSOR = onnxruntime.InferenceSession(model_path, providers = facefusion.globals.execution_providers)
	# 返回FRAME_PROCESSOR
	return FRAME_PROCESSOR


def clear_frame_processor() -> None:
	global FRAME_PROCESSOR

	FRAME_PROCESSOR = None


def get_model_matrix() -> Any:
	global MODEL_MATRIX

	with THREAD_LOCK:
		if MODEL_MATRIX is None:
			model_path = get_options('model').get('path')
			model = onnx.load(model_path)
			MODEL_MATRIX = numpy_helper.to_array(model.graph.initializer[-1])
	return MODEL_MATRIX


def clear_model_matrix() -> None:
	global MODEL_MATRIX

	MODEL_MATRIX = None


def get_options(key : Literal['model']) -> Any:
	global OPTIONS

	if OPTIONS is None:
		OPTIONS =\
		{
			'model': MODELS[frame_processors_globals.face_swapper_model]
		}
	return OPTIONS.get(key)


def set_options(key : Literal['model'], value : Any) -> None:
	global OPTIONS

	OPTIONS[key] = value


def register_args(program : ArgumentParser) -> None:
	program.add_argument('--face-swapper-model', help = wording.get('frame_processor_model_help'), dest = 'face_swapper_model', default = 'inswapper_128', choices = frame_processors_choices.face_swapper_models)


def apply_args() -> None:
	frame_processors_globals.face_swapper_model = 'inswapper_128' # args.face_swapper_model
	facefusion.globals.face_recognizer_model = 'arcface_inswapper'



# 定义一个函数pre_check，返回值为布尔值
def pre_check() -> bool:
	# 如果skip_download为False
	if not facefusion.globals.skip_download:
		# 获取相对路径'../.assets/models'
		download_directory_path = resolve_relative_path('../.assets/models')
		# 获取get_options('model')的url
		model_url = get_options('model').get('url')
		# 调用conditional_download函数，传入download_directory_path和[ model_url ]
		conditional_download(download_directory_path, [ model_url ])
	# 返回True
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
	if not is_image(facefusion.globals.source_path):
		update_status(wording.get('select_image_source') + wording.get('exclamation_mark'), NAME)
		return False
	elif not get_one_face(read_static_image(facefusion.globals.source_path)):
		update_status(wording.get('no_source_face_detected') + wording.get('exclamation_mark'), NAME)
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
	clear_model_matrix()
	clear_face_analyser()
	clear_content_analyser()
	read_static_image.cache_clear()


# 定义一个函数swap_face，用于交换source_face和target_face，并返回temp_frame
# 参数：source_face：Face类型；target_face：Face类型；temp_frame：Frame类型
# 返回值：Frame类型
def swap_face(source_face : Face, target_face : Face, temp_frame : Frame) -> Frame:
	# 获取帧处理器
	frame_processor = get_frame_processor()
	# 获取模型模板
	model_template = get_options('model').get('template')
	# 获取模型大小
	model_size = get_options('model').get('size')
	# 获取模型类型
	model_type = get_options('model').get('type')
	# 对帧进行位移，并获取位移矩阵
	crop_frame, affine_matrix = warp_face(temp_frame, target_face.kps, model_template, model_size)
	# 准备裁剪帧
	crop_frame = prepare_crop_frame(crop_frame)
	# 初始化帧处理器输入
	frame_processor_inputs = {}
	# 遍历帧处理器输入
	for frame_processor_input in frame_processor.get_inputs():
		# 如果输入名为source
		if frame_processor_input.name == 'source':
			# 如果模型类型为blendswap
			if model_type == 'blendswap':
				# 准备源帧
				frame_processor_inputs[frame_processor_input.name] = prepare_source_frame(source_face)
			# 否则
			else:
				# 准备源嵌入
				frame_processor_inputs[frame_processor_input.name] = prepare_source_embedding(source_face)
		# 如果输入名为target
		if frame_processor_input.name == 'target':
			# 将裁剪帧作为输入
			frame_processor_inputs[frame_processor_input.name] = crop_frame
	# 运行帧处理器，并获取裁剪帧
	crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
	# 归一化裁剪帧
	crop_frame = normalize_crop_frame(crop_frame)
	# 将裁剪帧粘贴到源帧上，并返回源帧
	temp_frame = paste_back(temp_frame, crop_frame, affine_matrix, facefusion.globals.face_mask_blur, facefusion.globals.face_mask_padding)
	return temp_frame


# 定义一个函数，用于准备源帧，参数为源脸
def prepare_source_frame(source_face : Face) -> numpy.ndarray[Any, Any]:
	# 读取静态图像
	source_frame = read_static_image(facefusion.globals.source_path)
	# 将源脸的kps应用到源帧上，并将其转换为112*112
	source_frame, _ = warp_face(source_frame, source_face.kps, 'arcface_v2', (112, 112))
	# 将源帧的像素值转换为[0,1]
	source_frame = source_frame[:, :, ::-1] / 255.0
	# 将源帧的维度转换为[2,0,1]
	source_frame = source_frame.transpose(2, 0, 1)
	# 将源帧增加一个维度，并转换为float32类型
	source_frame = numpy.expand_dims(source_frame, axis = 0).astype(numpy.float32)
	# 返回源帧
	return source_frame

def prepare_source_embedding(source_face : Face) -> Embedding:
	model_type = get_options('model').get('type')
	if model_type == 'inswapper':
		model_matrix = get_model_matrix()
		source_embedding = source_face.embedding.reshape((1, -1))
		source_embedding = numpy.dot(source_embedding, model_matrix) / numpy.linalg.norm(source_embedding)
	else:
		source_embedding = source_face.normed_embedding.reshape(1, -1)
	return source_embedding


def prepare_crop_frame(crop_frame : Frame) -> Frame:
	model_mean = get_options('model').get('mean')
	model_standard_deviation = get_options('model').get('standard_deviation')
	crop_frame = crop_frame[:, :, ::-1] / 255.0
	crop_frame = (crop_frame - model_mean) / model_standard_deviation
	crop_frame = crop_frame.transpose(2, 0, 1)
	crop_frame = numpy.expand_dims(crop_frame, axis = 0).astype(numpy.float32)
	return crop_frame


def normalize_crop_frame(crop_frame : Frame) -> Frame:
	crop_frame = crop_frame.transpose(1, 2, 0)
	crop_frame = (crop_frame * 255.0).round()
	crop_frame = crop_frame[:, :, ::-1].astype(numpy.uint8)
	return crop_frame


def process_frame(source_face : Face, reference_face : Face, temp_frame : Frame) -> Frame:
	if 'reference' in facefusion.globals.face_selector_mode:
		similar_faces = find_similar_faces(temp_frame, reference_face, facefusion.globals.reference_face_distance)
		if similar_faces:
			for similar_face in similar_faces:
				temp_frame = swap_face(source_face, similar_face, temp_frame)
	if 'one' in facefusion.globals.face_selector_mode:
		target_face = get_one_face(temp_frame)
		if target_face:
			temp_frame = swap_face(source_face, target_face, temp_frame)
	if 'many' in facefusion.globals.face_selector_mode:
		many_faces = get_many_faces(temp_frame)
		if many_faces:
			for target_face in many_faces:
				temp_frame = swap_face(source_face, target_face, temp_frame)
	return temp_frame


def process_frames(source_path : str, temp_frame_paths : List[str], update_progress : Update_Process) -> None:
	source_face = get_one_face(read_static_image(source_path))
	reference_face = get_face_reference() if 'reference' in facefusion.globals.face_selector_mode else None
	for temp_frame_path in temp_frame_paths:
		temp_frame = read_image(temp_frame_path)
		result_frame = process_frame(source_face, reference_face, temp_frame)
		write_image(temp_frame_path, result_frame)
		update_progress()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	source_face = get_one_face(read_static_image(source_path))
	target_frame = read_static_image(target_path)
	reference_face = get_one_face(target_frame, facefusion.globals.reference_face_position) if 'reference' in facefusion.globals.face_selector_mode else None
	result_frame = process_frame(source_face, reference_face, target_frame)
	write_image(output_path, result_frame)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	frame_processors.multi_process_frames(source_path, temp_frame_paths, process_frames)



if __name__ == '__main__':
	source_path = ".assets/examples/source.jpg"
	temp_frame_paths = ['/tmp/facefusion/target-1080p/0001.jpg', '/tmp/facefusion/target-1080p/0002.jpg', '/tmp/facefusion/target-1080p/0003.jpg', '/tmp/facefusion/target-1080p/0004.jpg', '/tmp/facefusion/target-1080p/0005.jpg', '/tmp/facefusion/target-1080p/0006.jpg', '/tmp/facefusion/target-1080p/0007.jpg', '/tmp/facefusion/target-1080p/0008.jpg', '/tmp/facefusion/target-1080p/0009.jpg', '/tmp/facefusion/target-1080p/0010.jpg']
	print("========开始处理")
	process_video(source_path, temp_frame_paths)