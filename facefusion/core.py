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


# def cli() -> None:
# 	signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
# 	program = ArgumentParser(formatter_class = lambda prog: HelpFormatter(prog, max_help_position = 120), add_help = False)
# 	# # general
# 	# program.add_argument('-s', '--source', help = wording.get('source_help'), dest = 'source_path')
# 	# program.add_argument('-t', '--target', help = wording.get('target_help'), dest = 'target_path')
# 	# program.add_argument('-o', '--output', help = wording.get('output_help'), dest = 'output_path')
# 	# program.add_argument('-v', '--version', version = metadata.get('name') + ' ' + metadata.get('version'), action = 'version')
# 	# # misc
# 	# group_misc = program.add_argument_group('misc')
# 	# group_misc.add_argument('--skip-download', help = wording.get('skip_download_help'), dest = 'skip_download', action = 'store_true')
# 	# group_misc.add_argument('--headless', help = wording.get('headless_help'), dest = 'headless', action = 'store_true')
# 	# # execution
# 	# group_execution = program.add_argument_group('execution')
# 	# group_execution.add_argument('--execution-providers', help = wording.get('execution_providers_help'), dest = 'execution_providers', default = [ 'cpu' ], choices = encode_execution_providers(onnxruntime.get_available_providers()), nargs = '+')
# 	# group_execution.add_argument('--execution-thread-count', help = wording.get('execution_thread_count_help'), dest = 'execution_thread_count', type = int, default = 4, choices = facefusion.choices.execution_thread_count_range, metavar = create_metavar(facefusion.choices.execution_thread_count_range))
# 	# group_execution.add_argument('--execution-queue-count', help = wording.get('execution_queue_count_help'), dest = 'execution_queue_count', type = int, default = 1, choices = facefusion.choices.execution_queue_count_range, metavar = create_metavar(facefusion.choices.execution_queue_count_range))
# 	# group_execution.add_argument('--max-memory', help = wording.get('max_memory_help'), dest = 'max_memory', type = int, choices = facefusion.choices.max_memory_range, metavar = create_metavar(facefusion.choices.max_memory_range))
# 	# # face analyser
# 	# group_face_analyser = program.add_argument_group('face analyser')
# 	# group_face_analyser.add_argument('--face-analyser-order', help = wording.get('face_analyser_order_help'), dest = 'face_analyser_order', default = 'left-right', choices = facefusion.choices.face_analyser_orders)
# 	# group_face_analyser.add_argument('--face-analyser-age', help = wording.get('face_analyser_age_help'), dest = 'face_analyser_age', choices = facefusion.choices.face_analyser_ages)
# 	# group_face_analyser.add_argument('--face-analyser-gender', help = wording.get('face_analyser_gender_help'), dest = 'face_analyser_gender', choices = facefusion.choices.face_analyser_genders)
# 	# group_face_analyser.add_argument('--face-detector-model', help = wording.get('face_detector_model_help'), dest = 'face_detector_model', default = 'retinaface', choices = facefusion.choices.face_detector_models)
# 	# group_face_analyser.add_argument('--face-detector-size', help = wording.get('face_detector_size_help'), dest = 'face_detector_size', default = '640x640', choices = facefusion.choices.face_detector_sizes)
# 	# group_face_analyser.add_argument('--face-detector-score', help = wording.get('face_detector_score_help'), dest = 'face_detector_score', type = float, default = 0.5, choices = facefusion.choices.face_detector_score_range, metavar = create_metavar(facefusion.choices.face_detector_score_range))
# 	# # face selector
# 	# group_face_selector = program.add_argument_group('face selector')
# 	# group_face_selector.add_argument('--face-selector-mode', help = wording.get('face_selector_mode_help'), dest = 'face_selector_mode', default = 'reference', choices = facefusion.choices.face_selector_modes)
# 	# group_face_selector.add_argument('--reference-face-position', help = wording.get('reference_face_position_help'), dest = 'reference_face_position', type = int, default = 0)
# 	# group_face_selector.add_argument('--reference-face-distance', help = wording.get('reference_face_distance_help'), dest = 'reference_face_distance', type = float, default = 0.6, choices = facefusion.choices.reference_face_distance_range, metavar = create_metavar(facefusion.choices.reference_face_distance_range))
# 	# group_face_selector.add_argument('--reference-frame-number', help = wording.get('reference_frame_number_help'), dest = 'reference_frame_number', type = int, default = 0)
# 	# # face mask
# 	# group_face_mask = program.add_argument_group('face mask')
# 	# group_face_mask.add_argument('--face-mask-blur', help = wording.get('face_mask_blur_help'), dest = 'face_mask_blur', type = float, default = 0.3, choices = facefusion.choices.face_mask_blur_range, metavar = create_metavar(facefusion.choices.face_mask_blur_range))
# 	# group_face_mask.add_argument('--face-mask-padding', help = wording.get('face_mask_padding_help'), dest = 'face_mask_padding', type = int, default = [ 0, 0, 0, 0 ], nargs = '+')
# 	# # frame extraction
# 	# group_frame_extraction = program.add_argument_group('frame extraction')
# 	# group_frame_extraction.add_argument('--trim-frame-start', help = wording.get('trim_frame_start_help'), dest = 'trim_frame_start', type = int)
# 	# group_frame_extraction.add_argument('--trim-frame-end', help = wording.get('trim_frame_end_help'), dest = 'trim_frame_end', type = int, default = 10)
# 	# group_frame_extraction.add_argument('--temp-frame-format', help = wording.get('temp_frame_format_help'), dest = 'temp_frame_format', default = 'jpg', choices = facefusion.choices.temp_frame_formats)
# 	# group_frame_extraction.add_argument('--temp-frame-quality', help = wording.get('temp_frame_quality_help'), dest = 'temp_frame_quality', type = int, default = 100, choices = facefusion.choices.temp_frame_quality_range, metavar = create_metavar(facefusion.choices.temp_frame_quality_range))
# 	# group_frame_extraction.add_argument('--keep-temp', help = wording.get('keep_temp_help'), dest = 'keep_temp', action = 'store_true')
# 	# # output creation
# 	# group_output_creation = program.add_argument_group('output creation')
# 	# group_output_creation.add_argument('--output-image-quality', help = wording.get('output_image_quality_help'), dest = 'output_image_quality', type = int, default = 80, choices = facefusion.choices.output_image_quality_range, metavar = create_metavar(facefusion.choices.output_image_quality_range))
# 	# group_output_creation.add_argument('--output-video-encoder', help = wording.get('output_video_encoder_help'), dest = 'output_video_encoder', default = 'libx264', choices = facefusion.choices.output_video_encoders)
# 	# group_output_creation.add_argument('--output-video-quality', help = wording.get('output_video_quality_help'), dest = 'output_video_quality', type = int, default = 80, choices = facefusion.choices.output_video_quality_range, metavar = create_metavar(facefusion.choices.output_video_quality_range))
# 	# group_output_creation.add_argument('--keep-fps', help = wording.get('keep_fps_help'), dest = 'keep_fps', action = 'store_true')
# 	# group_output_creation.add_argument('--skip-audio', help = wording.get('skip_audio_help'), dest = 'skip_audio', action = 'store_true')
# 	# # frame processors
# 	# available_frame_processors = list_module_names('facefusion/processors/frame/modules')
# 	# program = ArgumentParser(parents = [ program ], formatter_class = program.formatter_class, add_help = True)
# 	# group_frame_processors = program.add_argument_group('frame processors')
# 	# group_frame_processors.add_argument('--frame-processors', help = wording.get('frame_processors_help').format(choices = ', '.join(available_frame_processors)), dest = 'frame_processors', default = [ 'face_swapper','face_enhancer' ], nargs = '+')
# 	# for frame_processor in available_frame_processors:
# 	# 	frame_processor_module = load_frame_processor_module(frame_processor)
# 	# 	frame_processor_module.register_args(group_frame_processors)
# 	# # uis
# 	# group_uis = program.add_argument_group('uis')
# 	# group_uis.add_argument('--ui-layouts', help = wording.get('ui_layouts_help').format(choices = ', '.join(list_module_names('facefusion/uis/layouts'))), dest = 'ui_layouts', default = [ 'default' ], nargs = '+')
# 	run(program)


# def apply_args(program : ArgumentParser) -> None:
# 	args = program.parse_args()
# 	# # general
# 	# facefusion.globals.source_path = ".assets/examples/source.jpg"
# 	# facefusion.globals.target_path = ".assets/examples/target-1080p.mp4"
# 	# facefusion.globals.output_path = ".assets/examples"
# 	# # misc
# 	# facefusion.globals.skip_download = args.skip_download
# 	# facefusion.globals.headless = args.headless
# 	# # execution
# 	# facefusion.globals.execution_providers = decode_execution_providers(args.execution_providers)
# 	# facefusion.globals.execution_thread_count = args.execution_thread_count
# 	# facefusion.globals.execution_queue_count = args.execution_queue_count
# 	# facefusion.globals.max_memory = args.max_memory
# 	# # face analyser
# 	# facefusion.globals.face_analyser_order = args.face_analyser_order
# 	# facefusion.globals.face_analyser_age = args.face_analyser_age
# 	# facefusion.globals.face_analyser_gender = args.face_analyser_gender
# 	# facefusion.globals.face_detector_model = args.face_detector_model
# 	# facefusion.globals.face_detector_size = args.face_detector_size
# 	# facefusion.globals.face_detector_score = args.face_detector_score
# 	# # face selector
# 	# facefusion.globals.face_selector_mode = args.face_selector_mode
# 	# facefusion.globals.reference_face_position = args.reference_face_position
# 	# facefusion.globals.reference_face_distance = args.reference_face_distance
# 	# facefusion.globals.reference_frame_number = args.reference_frame_number
# 	# # face mask
# 	# facefusion.globals.face_mask_blur = args.face_mask_blur
# 	# facefusion.globals.face_mask_padding = normalize_padding(args.face_mask_padding)
# 	# print(args.face_mask_padding)
# 	# print(facefusion.globals.face_mask_padding)
# 	# # frame extraction
# 	# facefusion.globals.trim_frame_start = args.trim_frame_start
# 	# facefusion.globals.trim_frame_end = args.trim_frame_end
# 	# facefusion.globals.temp_frame_format = args.temp_frame_format
# 	# facefusion.globals.temp_frame_quality = args.temp_frame_quality
# 	# facefusion.globals.keep_temp = args.keep_temp
# 	# output creation
# 	# facefusion.globals.output_image_quality = args.output_image_quality
# 	# print(args.output_image_quality)
# 	# facefusion.globals.output_video_encoder = args.output_video_encoder
# 	# print(args.output_video_encoder)
# 	# facefusion.globals.output_video_quality = args.output_video_quality
# 	# print(args.output_video_quality)
# 	# facefusion.globals.keep_fps = args.keep_fps
# 	# facefusion.globals.skip_audio = args.skip_audio
# 	# frame processors
# 	available_frame_processors = list_module_names('facefusion/processors/frame/modules')
# 	print(available_frame_processors)
# 	print(args.frame_processors)
# 	# facefusion.globals.frame_processors = args.frame_processors
# 	for frame_processor in available_frame_processors:
# 		frame_processor_module = load_frame_processor_module(frame_processor)
# 		frame_processor_module.apply_args(program)
# 	# uis
# 	# facefusion.globals.ui_layouts = args.ui_layouts


# 定义一个函数run，参数为ArgumentParser，返回值为None
def run() -> None:
	# 调用apply_args函数加载参数
	# print(f"加载参数：")
	# print(program.parse_args())
	# print("上述参数")
	# apply_args(program)

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
	pass