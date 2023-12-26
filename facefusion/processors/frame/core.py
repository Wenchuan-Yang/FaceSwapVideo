import sys
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List
from tqdm import tqdm

import facefusion.globals
from facefusion.typing import Process_Frames
from facefusion import wording
from facefusion.utilities import encode_execution_providers

FRAME_PROCESSORS_MODULES : List[ModuleType] = [] # frame processors modules
# prame processors methods
FRAME_PROCESSORS_METHODS =\
[
	'get_frame_processor',
	'clear_frame_processor',
	'get_options',
	'set_options',
	'register_args',
	'apply_args',
	'pre_check',
	'pre_process',
	'process_frame',
	'process_frames',
	'process_image',
	'process_video',
	'post_process'
]


# 加载帧处理器模块
def load_frame_processor_module(frame_processor : str) -> Any:
	'''
	加载帧处理器模块
	:param frame_processor: 帧处理器名称
	:return: 帧处理器模块
	'''
	# 尝试导入frame_processor模块
	try:
		frame_processor_module = importlib.import_module('facefusion.processors.frame.modules.' + frame_processor)
		# 检查frame_processor模块是否包含FRAME_PROCESSORS_METHODS中的方法
		for method_name in FRAME_PROCESSORS_METHODS:
			if not hasattr(frame_processor_module, method_name):
				raise NotImplementedError
	# 如果模块没有找到，则退出
	except ModuleNotFoundError:
		sys.exit(wording.get('frame_processor_not_loaded').format(frame_processor = frame_processor))
	# 如果模块没有实现FRAME_PROCESSORS_METHODS中的方法，则退出
	except NotImplementedError:
		sys.exit(wording.get('frame_processor_not_implemented').format(frame_processor = frame_processor))
	# 返回frame_processor模块
	return frame_processor_module


# 定义一个函数，用于获取帧处理器模块列表
def get_frame_processors_modules(frame_processors : List[str]) -> List[ModuleType]:
	# 声明一个全局变量，用于存储帧处理器模块列表
	global FRAME_PROCESSORS_MODULES

	# 如果帧处理器模块列表为空
	if not FRAME_PROCESSORS_MODULES:
		# 遍历帧处理器列表
		for frame_processor in frame_processors:
			# 加载帧处理器模块
			frame_processor_module = load_frame_processor_module(frame_processor)
			# 将帧处理器模块添加到帧处理器模块列表中
			FRAME_PROCESSORS_MODULES.append(frame_processor_module)
	# 返回帧处理器模块列表
	return FRAME_PROCESSORS_MODULES


# 定义一个函数，用于清空帧处理器模块
def clear_frame_processors_modules() -> None:
	# 声明一个全局变量，用于存储帧处理器模块
	global FRAME_PROCESSORS_MODULES

	# 遍历获取的帧处理器模块
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		# 调用帧处理器模块的clear_frame_processor函数
		frame_processor_module.clear_frame_processor()
	# 将FRAME_PROCESSORS_MODULES置空
	FRAME_PROCESSORS_MODULES = []


# 定义一个多进程处理帧的函数，参数为源路径、临时帧路径列表和处理帧函数，无返回值
def multi_process_frames(source_path : str, temp_frame_paths : List[str], process_frames : Process_Frames) -> None:
	# 使用tqdm模块创建进度条，总帧数为temp_frame_paths的长度，描述为wording.get('processing')，单位为frame，ascii为' ='
	with tqdm(total = len(temp_frame_paths), desc = wording.get('processing'), unit = 'frame', ascii = ' =') as progress:
		# total: 表示进度条的总长度
		# desc: 给进度条添加的描述前缀
		# unit: 指定每次进度条更新的单位
		# ascii: 使用ASCII字符来表示进度条，而非默认的Unicode字符
	
		# 设置进度条后缀，包含execution_providers、execution_thread_count、execution_queue_count
		progress.set_postfix(
		{
			'execution_providers': encode_execution_providers(facefusion.globals.execution_providers),
			'execution_thread_count': facefusion.globals.execution_thread_count,
			'execution_queue_count': facefusion.globals.execution_queue_count
		})
		# 使用ThreadPoolExecutor创建线程池，最大线程数为facefusion.globals.execution_thread_count
		with ThreadPoolExecutor(max_workers = facefusion.globals.execution_thread_count) as executor:
			# 创建一个空列表，用于存放futures
			futures = []
			# 使用create_queue函数创建一个队列，用于存放temp_frame_paths
			queue_temp_frame_paths : Queue[str] = create_queue(temp_frame_paths)
			# 计算每个future的队列长度，取最大值为temp_frame_paths的长度除以facefusion.globals.execution_thread_count乘以facefusion.globals.execution_queue_count，最小值为1
			queue_per_future = max(len(temp_frame_paths) // facefusion.globals.execution_thread_count * facefusion.globals.execution_queue_count, 1)
			# 当queue_temp_frame_paths不为空时，循环取出队列中的元素，并提交到线程池中，将取出的元素放入futures中
			while not queue_temp_frame_paths.empty():
				payload_temp_frame_paths = pick_queue(queue_temp_frame_paths, queue_per_future)
				future = executor.submit(process_frames, source_path, payload_temp_frame_paths, progress.update)
				futures.append(future)
			# 当futures中的每一个future都完成时，返回结果
			for future_done in as_completed(futures):
				future_done.result()


# 定义一个函数，用于创建一个队列，参数是一个临时帧路径列表
def create_queue(temp_frame_paths : List[str]) -> Queue[str]:
	# 创建一个队列
	queue : Queue[str] = Queue()
	# 遍历临时帧路径列表
	for frame_path in temp_frame_paths:
		# 将每个帧路径添加到队列中
		queue.put(frame_path)
	# 返回队列
	return queue


# 定义一个函数，用于从队列中选择队列，参数是一个队列和一个每个未来队列的数量
def pick_queue(queue : Queue[str], queue_per_future : int) -> List[str]:
	# 创建一个列表，用于存放选择的队列
	queues = []
	# 遍历每个未来队列的数量
	for _ in range(queue_per_future):
		# 如果队列不为空
		if not queue.empty():
			# 从队列中取出一个队列
			queues.append(queue.get())
	# 返回选择的队列列表
	return queues
