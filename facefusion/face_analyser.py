from typing import Any, Optional, List, Dict, Tuple
import threading
import cv2
import numpy
import onnxruntime

import facefusion.globals
from facefusion.face_cache import get_faces_cache, set_faces_cache
from facefusion.face_helper import warp_face, create_static_anchors, distance_to_kps, distance_to_bbox, apply_nms
from facefusion.typing import Frame, Face, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, ModelValue, Bbox, Kps, Score, Embedding
from facefusion.utilities import resolve_relative_path, conditional_download
from facefusion.vision import resize_frame_dimension

FACE_ANALYSER = None # 人脸分析
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()
MODELS : Dict[str, ModelValue] =\
{
	'face_detector_retinaface': # 人脸检测模型
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx',
		'path': resolve_relative_path('../.assets/models/retinaface_10g.onnx')
	},
	'face_detector_yunet': # 人脸检测模型
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/yunet_2023mar.onnx',
		'path': resolve_relative_path('../.assets/models/yunet_2023mar.onnx')
	},
	'face_recognizer_arcface_blendswap': # 人脸识别模型
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
	},
	'face_recognizer_arcface_inswapper': # 人脸识别模型
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
	},
	'face_recognizer_arcface_simswap': # 人脸识别模型
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_simswap.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_simswap.onnx')
	},
	'gender_age': # 性别年龄识别
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gender_age.onnx',
		'path': resolve_relative_path('../.assets/models/gender_age.onnx')
	}
}


def get_face_analyser() -> Any:
	global FACE_ANALYSER

	with THREAD_LOCK:
		if FACE_ANALYSER is None:
			# 加载人脸检测模型
			if facefusion.globals.face_detector_model == 'retinaface': 
				face_detector = onnxruntime.InferenceSession(MODELS.get('face_detector_retinaface').get('path'), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_detector_model == 'yunet':
				face_detector = cv2.FaceDetectorYN.create(MODELS.get('face_detector_yunet').get('path'), '', (0, 0))

			# 加载人脸识别模型
			if facefusion.globals.face_recognizer_model == 'arcface_blendswap':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_blendswap').get('path'), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_recognizer_model == 'arcface_inswapper':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_inswapper').get('path'), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_recognizer_model == 'arcface_simswap':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_simswap').get('path'), providers = facefusion.globals.execution_providers)
			gender_age = onnxruntime.InferenceSession(MODELS.get('gender_age').get('path'), providers = facefusion.globals.execution_providers)
			FACE_ANALYSER =\
			{
				'face_detector': face_detector,
				'face_recognizer': face_recognizer,
				'gender_age': gender_age
			}
	return FACE_ANALYSER


def clear_face_analyser() -> Any:
	global FACE_ANALYSER

	FACE_ANALYSER = None


# 下载模型
# 定义一个函数，返回布尔值
def pre_check() -> bool:
	# 如果skip_download不为1，则执行if语句
	if not facefusion.globals.skip_download:
		# 获取相对路径
		download_directory_path = resolve_relative_path('../.assets/models')
		# 获取模型url
		model_urls =\
		[
			MODELS.get('face_detector_retinaface').get('url'),
			MODELS.get('face_detector_yunet').get('url'),
			MODELS.get('face_recognizer_arcface_inswapper').get('url'),
			MODELS.get('face_recognizer_arcface_simswap').get('url'),
			MODELS.get('gender_age').get('url')
		]
		# 条件下载
		conditional_download(download_directory_path, model_urls)
	# 返回True
	return True


# 定义一个函数，用于提取帧中的脸部
def extract_faces(frame: Frame) -> List[Face]:
	# 获取人脸检测模型的宽度和高度
	face_detector_width, face_detector_height = map(int, facefusion.globals.face_detector_size.split('x'))
	# 获取帧的高度、宽度、通道数
	frame_height, frame_width, _ = frame.shape
	# 将帧按照模型的大小进行缩放
	temp_frame = resize_frame_dimension(frame, face_detector_width, face_detector_height)
	# 获取缩放后的帧的高度、宽度、通道数
	temp_frame_height, temp_frame_width, _ = temp_frame.shape
	# 计算缩放比例
	ratio_height = frame_height / temp_frame_height
	ratio_width = frame_width / temp_frame_width
	# 根据模型类型，使用不同的模型进行检测
	if facefusion.globals.face_detector_model == 'retinaface':
		# 使用retinaface模型进行检测
		bbox_list, kps_list, score_list = detect_with_retinaface(temp_frame, temp_frame_height, temp_frame_width, face_detector_height, face_detector_width, ratio_height, ratio_width)
		# 根据检测结果创建脸部
		return create_faces(frame, bbox_list, kps_list, score_list)
	elif facefusion.globals.face_detector_model == 'yunet':
		# 使用yunet模型进行检测
		bbox_list, kps_list, score_list = detect_with_yunet(temp_frame, temp_frame_height, temp_frame_width, ratio_height, ratio_width)
		# 根据检测结果创建脸部
		return create_faces(frame, bbox_list, kps_list, score_list)
	# 如果没有检测到脸部，则返回空列表
	return []


def detect_with_retinaface(temp_frame : Frame, temp_frame_height : int, temp_frame_width : int, face_detector_height : int, face_detector_width : int, ratio_height : float, ratio_width : float) -> Tuple[List[Bbox], List[Kps], List[Score]]:
	face_detector = get_face_analyser().get('face_detector')
	bbox_list = []
	kps_list = []
	score_list = []
	feature_strides = [ 8, 16, 32 ]
	feature_map_channel = 3
	anchor_total = 2
	prepare_frame = numpy.zeros((face_detector_height, face_detector_width, 3))
	prepare_frame[:temp_frame_height, :temp_frame_width, :] = temp_frame
	temp_frame = (prepare_frame - 127.5) / 128.0
	temp_frame = numpy.expand_dims(temp_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	with THREAD_SEMAPHORE:
		detections = face_detector.run(None,
		{
			face_detector.get_inputs()[0].name: temp_frame
		})
	for index, feature_stride in enumerate(feature_strides):
		keep_indices = numpy.where(detections[index] >= facefusion.globals.face_detector_score)[0]
		if keep_indices.any():
			stride_height = face_detector_height // feature_stride
			stride_width = face_detector_width // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
			bbox_raw = (detections[index + feature_map_channel] * feature_stride)
			kps_raw = detections[index + feature_map_channel * 2] * feature_stride
			for bbox in distance_to_bbox(anchors, bbox_raw)[keep_indices]:
				bbox_list.append(numpy.array(
				[
					bbox[0] * ratio_width,
					bbox[1] * ratio_height,
					bbox[2] * ratio_width,
					bbox[3] * ratio_height
				]))
			for kps in distance_to_kps(anchors, kps_raw)[keep_indices]:
				kps_list.append(kps * [ ratio_width, ratio_height ])
			for score in detections[index][keep_indices]:
				score_list.append(score[0])
	return bbox_list, kps_list, score_list


def detect_with_yunet(temp_frame : Frame, temp_frame_height : int, temp_frame_width : int, ratio_height : float, ratio_width : float) -> Tuple[List[Bbox], List[Kps], List[Score]]:
	face_detector = get_face_analyser().get('face_detector')
	face_detector.setInputSize((temp_frame_width, temp_frame_height))
	face_detector.setScoreThreshold(facefusion.globals.face_detector_score)
	bbox_list = []
	kps_list = []
	score_list = []
	with THREAD_SEMAPHORE:
		_, detections = face_detector.detect(temp_frame)
	if detections.any():
		for detection in detections:
			bbox_list.append(numpy.array(
			[
				detection[0] * ratio_width,
				detection[1] * ratio_height,
				(detection[0] + detection[2]) * ratio_width,
				(detection[1] + detection[3]) * ratio_height
			]))
			kps_list.append(detection[4:14].reshape((5, 2)) * [ ratio_width, ratio_height])
			score_list.append(detection[14])
	return bbox_list, kps_list, score_list


# 定义一个函数，用于创建Face对象，参数为帧、边界框列表、关键点列表和分数列表，返回Face列表
def create_faces(frame : Frame, bbox_list : List[Bbox], kps_list : List[Kps], score_list : List[Score]) -> List[Face] :
	# 创建一个Face列表
	faces : List[Face] = []
	# 如果facefusion.globals.face_detector_score大于0
	if facefusion.globals.face_detector_score > 0:
		# 应用非极大值抑制，保留索引
		keep_indices = apply_nms(bbox_list, 0.4)
		# 遍历保留的索引
		for index in keep_indices:
			# 获取边界框
			bbox = bbox_list[index]
			# 获取关键点
			kps = kps_list[index]
			# 获取分数
			score = score_list[index]
			# 计算嵌入向量
			embedding, normed_embedding = calc_embedding(frame, kps)
			# 检测性别和年龄
			gender, age = detect_gender_age(frame, kps)
			# 将边界框、关键点、分数、嵌入向量、归一化嵌入向量、性别和年龄添加到Face列表中
			faces.append(Face(
				bbox = bbox,
				kps = kps,
				score = score,
				embedding = embedding,
				normed_embedding = normed_embedding, 
				gender = gender,
				age = age
			))
	# 返回Face列表
	return faces


# 定义一个函数，用于计算特征向量，参数为帧和关键点，返回值为特征向量
def calc_embedding(temp_frame : Frame, kps : Kps) -> Tuple[Embedding, Embedding]:
	# 获取人脸分析器
	face_recognizer = get_face_analyser().get('face_recognizer')
	# 对帧进行裁剪，并获取变换矩阵
	crop_frame, matrix = warp_face(temp_frame, kps, 'arcface_v2', (112, 112))
	# 将裁剪后的帧转换为float32类型，并除以127.5，再减去1
	crop_frame = crop_frame.astype(numpy.float32) / 127.5 - 1
	# 将裁剪后的帧转换为RGB格式，并转置
	crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
	# 将裁剪后的帧增加一个维度
	crop_frame = numpy.expand_dims(crop_frame, axis = 0)
	# 运行人脸识别器，获取特征向量
	embedding = face_recognizer.run(None,
	{
		face_recognizer.get_inputs()[0].name: crop_frame
	})[0]
	# 将特征向量拉平
	embedding = embedding.ravel()
	# 计算特征向量的范数
	normed_embedding = embedding / numpy.linalg.norm(embedding)
	# 返回特征向量和归一化后的特征向量
	return embedding, normed_embedding


# 定义一个函数，检测性别和年龄，参数为帧和关键点
def detect_gender_age(frame : Frame, kps : Kps) -> Tuple[int, int]:
	# 获取性别年龄分析器
	gender_age = get_face_analyser().get('gender_age')
	# 对帧进行裁剪，并获取仿射矩阵
	crop_frame, affine_matrix = warp_face(frame, kps, 'arcface_v2', (96, 96))
	# 将裁剪后的帧扩展维度，并转换类型
	crop_frame = numpy.expand_dims(crop_frame, axis = 0).transpose(0, 3, 1, 2).astype(numpy.float32)
	# 运行性别年龄分析器，获取预测结果
	prediction = gender_age.run(None,
	{
		gender_age.get_inputs()[0].name: crop_frame
	})[0][0]
	# 获取性别，并转换为整型
	gender = int(numpy.argmax(prediction[:2]))
	# 获取年龄，并转换为整型，保留两位小数
	age = int(numpy.round(prediction[2] * 100))
	# 返回性别和年龄
	return gender, age


def get_one_face(frame : Frame, position : int = 0) -> Optional[Face]:
	many_faces = get_many_faces(frame)
	if many_faces:
		try:
			return many_faces[position]
		except IndexError:
			return many_faces[-1]
	return None


def get_many_faces(frame : Frame) -> List[Face]:
	try:
		faces_cache = get_faces_cache(frame)
		if faces_cache:
			faces = faces_cache
		else:
			faces = extract_faces(frame)
			set_faces_cache(frame, faces)
		if facefusion.globals.face_analyser_order:
			faces = sort_by_order(faces, facefusion.globals.face_analyser_order)
		if facefusion.globals.face_analyser_age:
			faces = filter_by_age(faces, facefusion.globals.face_analyser_age)
		if facefusion.globals.face_analyser_gender:
			faces = filter_by_gender(faces, facefusion.globals.face_analyser_gender)
		return faces
	except (AttributeError, ValueError):
		return []


def find_similar_faces(frame : Frame, reference_face : Face, face_distance : float) -> List[Face]:
	many_faces = get_many_faces(frame)
	similar_faces = []
	if many_faces:
		for face in many_faces:
			if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
				current_face_distance = 1 - numpy.dot(face.normed_embedding, reference_face.normed_embedding)
				if current_face_distance < face_distance:
					similar_faces.append(face)
	return similar_faces


def sort_by_order(faces : List[Face], order : FaceAnalyserOrder) -> List[Face]:
	if order == 'left-right':
		return sorted(faces, key = lambda face: face.bbox[0])
	if order == 'right-left':
		return sorted(faces, key = lambda face: face.bbox[0], reverse = True)
	if order == 'top-bottom':
		return sorted(faces, key = lambda face: face.bbox[1])
	if order == 'bottom-top':
		return sorted(faces, key = lambda face: face.bbox[1], reverse = True)
	if order == 'small-large':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
	if order == 'large-small':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse = True)
	if order == 'best-worst':
		return sorted(faces, key = lambda face: face.score, reverse = True)
	if order == 'worst-best':
		return sorted(faces, key = lambda face: face.score)
	return faces


def filter_by_age(faces : List[Face], age : FaceAnalyserAge) -> List[Face]:
	filter_faces = []
	for face in faces:
		if face.age < 13 and age == 'child':
			filter_faces.append(face)
		elif face.age < 19 and age == 'teen':
			filter_faces.append(face)
		elif face.age < 60 and age == 'adult':
			filter_faces.append(face)
		elif face.age > 59 and age == 'senior':
			filter_faces.append(face)
	return filter_faces


def filter_by_gender(faces : List[Face], gender : FaceAnalyserGender) -> List[Face]:
	filter_faces = []
	for face in faces:
		if face.gender == 0 and gender == 'female':
			filter_faces.append(face)
		if face.gender == 1 and gender == 'male':
			filter_faces.append(face)
	return filter_faces
