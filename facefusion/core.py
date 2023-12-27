import os

os.environ['OMP_NUM_THREADS'] = '1'

import signal
import sys
import warnings
import platform
import shutil
import onnxruntime
from argparse import ArgumentParser, HelpFormatter

import facefusion.choices
import facefusion.globals
from facefusion.face_analyser import get_one_face
from facefusion.face_reference import get_face_reference, set_face_reference
from facefusion.vision import get_video_frame, read_image
from facefusion import face_analyser, content_analyser, metadata, wording
from facefusion.content_analyser import analyse_image, analyse_video
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from facefusion.utilities import is_image, is_video, detect_fps, compress_image, merge_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clear_temp, list_module_names, encode_execution_providers, decode_execution_providers, normalize_output_path, normalize_padding, create_metavar, update_status

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')

# 定义一个函数run，参数为ArgumentParser，返回值为None
def run() -> None:
	available_frame_processors = list_module_names('facefusion/processors/frame/modules')
	# facefusion.globals.frame_processors = args.frame_processors
	print(f"available_frame_processors = {available_frame_processors}")
	for frame_processor in available_frame_processors:
		frame_processor_module = load_frame_processor_module(frame_processor)
		frame_processor_module.apply_args()

	# 调用limit_resources函数
	limit_resources()
	# 调用pre_check函数，如果返回值为False或者content_analyser.pre_check()或者face_analyser.pre_check()返回值为False，则返回
	if not pre_check() or not content_analyser.pre_check() or not face_analyser.pre_check():
		return
	# 遍历frame_processors_modules，如果返回值为False，则返回
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		print()
		print(f"Pre-checking {frame_processor_module}")
		if not frame_processor_module.pre_check():
			return
	# 如果headless为True，则调用conditional_process函数
	conditional_process()


def destroy() -> None:
	if facefusion.globals.target_path:
		clear_temp(facefusion.globals.target_path)
	sys.exit()


# 定义一个limit_resources函数，返回值为None
def limit_resources() -> None:
	# 如果facefusion.globals.max_memory存在，则将max_memory转换为字节
	if facefusion.globals.max_memory:
		memory = facefusion.globals.max_memory * 1024 ** 3
		# 如果操作系统为mac，则将memory转换为字节
		if platform.system().lower() == 'darwin':
			memory = facefusion.globals.max_memory * 1024 ** 6
		# 如果操作系统为windows，则导入ctypes模块，并设置内存
		if platform.system().lower() == 'windows':
			import ctypes
			kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
			kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
		# 如果操作系统为linux，则导入resource模块，并设置内存
		else:
			import resource
			resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
	if sys.version_info < (3, 9):
		update_status(wording.get('python_not_supported').format(version = '3.9'))
		return False
	if not shutil.which('ffmpeg'):
		update_status(wording.get('ffmpeg_not_installed'))
		return False
	return True


# 定义一个函数conditional_process，返回值为None
def conditional_process() -> None:
	# 调用conditional_set_face_reference函数
	conditional_set_face_reference()
	# 遍历frame_processors模块
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		# 如果pre_process函数返回False，则返回
		if not frame_processor_module.pre_process('output'):
			return
	# 判断target_path是否为图片
	if is_image(facefusion.globals.target_path):
		# 如果是图片，则调用process_image函数
		process_image()
	# 判断target_path是否为视频
	if is_video(facefusion.globals.target_path):
		# 如果是视频，则调用process_video函数
		process_video()


# 定义一个函数，用于设置参考人脸
def conditional_set_face_reference() -> None:
	# 如果facefusion.globals.face_selector_mode中的reference为真，且没有获取到参考人脸
	if 'reference' in facefusion.globals.face_selector_mode and not get_face_reference():
		# 如果facefusion.globals.target_path指向的是视频
		if is_video(facefusion.globals.target_path):
			# 获取视频帧
			print(f"========获取视频帧=========")
			reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
		else:
			# 读取图片
			reference_frame = read_image(facefusion.globals.target_path)
		# 获取参考人脸
		reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
		# 设置参考人脸
		set_face_reference(reference_face)


def process_image() -> None:
	if analyse_image(facefusion.globals.target_path):
		return
	shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
	# process frame
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		update_status(wording.get('processing'), frame_processor_module.NAME)
		frame_processor_module.process_image(facefusion.globals.source_path, facefusion.globals.output_path, facefusion.globals.output_path)
		frame_processor_module.post_process()
	# compress image
	update_status(wording.get('compressing_image'))
	if not compress_image(facefusion.globals.output_path):
		update_status(wording.get('compressing_image_failed'))
	# validate image
	if is_image(facefusion.globals.output_path):
		update_status(wording.get('processing_image_succeed'))
	else:
		update_status(wording.get('processing_image_failed'))


def process_video() -> None:
	if analyse_video(facefusion.globals.target_path, facefusion.globals.trim_frame_start, facefusion.globals.trim_frame_end):
		return
	fps = detect_fps(facefusion.globals.target_path) if facefusion.globals.keep_fps else 25.0
	print(facefusion.globals.keep_fps)
	print(f"fps = {fps}")

	# create temp
	# update_status(wording.get('creating_temp'))
	print(f"创建文件夹")
	create_temp(facefusion.globals.target_path)
	# extract frames
	update_status(wording.get('extracting_frames_fps').format(fps = fps))
	extract_frames(facefusion.globals.target_path, fps)
	# process frame
	temp_frame_paths = get_temp_frame_paths(facefusion.globals.target_path)
	print(f"temp_frame_paths = {temp_frame_paths}")
	if temp_frame_paths:
		for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
			update_status(wording.get('processing'), frame_processor_module.NAME)
			print()
			print(frame_processor_module)
			frame_processor_module.process_video(facefusion.globals.source_path, temp_frame_paths)
			frame_processor_module.post_process()
	else:
		update_status(wording.get('temp_frames_not_found'))
		return
	# merge video
	update_status(wording.get('merging_video_fps').format(fps = fps))
	if not merge_video(facefusion.globals.target_path, fps):
		update_status(wording.get('merging_video_failed'))
		return
	# handle audio
	if facefusion.globals.skip_audio:
		update_status(wording.get('skipping_audio'))
		move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
	else:
		update_status(wording.get('restoring_audio'))
		restore_audio_result = restore_audio(facefusion.globals.target_path, facefusion.globals.output_path)
		print(f"===========  恢复音频的结果{restore_audio_result}")
		if not restore_audio_result:
			update_status(wording.get('restoring_audio_failed'))
			move_temp(facefusion.globals.target_path, facefusion.globals.output_path)
	# clear temp
	update_status(wording.get('clearing_temp'))
	clear_temp(facefusion.globals.target_path)
	# validate video
	if is_video(facefusion.globals.output_path):
		update_status(wording.get('processing_video_succeed'))
	else:
		update_status(wording.get('processing_video_failed'))


if __name__ == "__main__":
	run()