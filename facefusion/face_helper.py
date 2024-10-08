from typing import Any, Dict, Tuple, List
from functools import lru_cache
from cv2.typing import Size
import cv2
import numpy

from facefusion.typing import Bbox, Kps, Frame, Matrix, Template, Padding

# templates
TEMPLATES : Dict[Template, numpy.ndarray[Any, Any]] =\
{
	'arcface_v1': numpy.array(
	[
		[ 39.7300, 51.1380 ],
		[ 72.2700, 51.1380 ],
		[ 56.0000, 68.4930 ],
		[ 42.4630, 87.0100 ],
		[ 69.5370, 87.0100 ]
	]),
	'arcface_v2': numpy.array(
	[
		[ 38.2946, 51.6963 ],
		[ 73.5318, 51.5014 ],
		[ 56.0252, 71.7366 ],
		[ 41.5493, 92.3655 ],
		[ 70.7299, 92.2041 ]
	]),
	'ffhq': numpy.array(
	[
		[ 192.98138, 239.94708 ],
		[ 318.90277, 240.1936 ],
		[ 256.63416, 314.01935 ],
		[ 201.26117, 371.41043 ],
		[ 313.08905, 371.15118 ]
	])
}


# 定义一个函数warp_face，用于对帧进行裁剪，并返回裁剪后的帧和仿射矩阵
# 参数：temp_frame：帧；kps：关键点；template：模板；size：尺寸
# 返回：裁剪后的帧和仿射矩阵
def warp_face(temp_frame : Frame, kps : Kps, template : Template, size : Size) -> Tuple[Frame, Matrix]:
	# 获取模板的归一化值
	normed_template = TEMPLATES.get(template) * size[1] / size[0]
	# 使用LMedS方法估计仿射矩阵
	affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method = cv2.LMEDS)[0]
	# 使用仿射矩阵对帧进行裁剪
	crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (size[1], size[1]), borderMode = cv2.BORDER_REPLICATE)
	# 返回裁剪后的帧和仿射矩阵
	return crop_frame, affine_matrix


# 定义一个函数，用于将裁剪后的帧粘贴到临时帧上
# 参数：temp_frame：临时帧；crop_frame：裁剪帧；affine_matrix：仿射矩阵；face_mask_blur：模糊掩码；face_mask_padding：填充掩码
# 返回：粘贴后的帧
def paste_back(temp_frame : Frame, crop_frame: Frame, affine_matrix : Matrix, face_mask_blur : float, face_mask_padding : Padding) -> Frame:
	# 计算仿射矩阵的反矩阵
	inverse_matrix = cv2.invertAffineTransform(affine_matrix)
	# 获取临时帧的大小
	temp_frame_size = temp_frame.shape[:2][::-1]
	# 获取裁剪帧的大小
	mask_size = tuple(crop_frame.shape[:2])
	# 创建静态掩码帧
	mask_frame = create_static_mask_frame(mask_size, face_mask_blur, face_mask_padding)
	# 将掩码帧应用到反矩阵上，并裁剪到0-1之间
	inverse_mask_frame = cv2.warpAffine(mask_frame, inverse_matrix, temp_frame_size).clip(0, 1)
	# 将裁剪帧应用到反矩阵上，并使用边界填充
	inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode = cv2.BORDER_REPLICATE)
	# 复制临时帧
	paste_frame = temp_frame.copy()
	# 将裁剪帧的RGB值 paste到临时帧上
	paste_frame[:, :, 0] = inverse_mask_frame * inverse_crop_frame[:, :, 0] + (1 - inverse_mask_frame) * temp_frame[:, :, 0]
	paste_frame[:, :, 1] = inverse_mask_frame * inverse_crop_frame[:, :, 1] + (1 - inverse_mask_frame) * temp_frame[:, :, 1]
	paste_frame[:, :, 2] = inverse_mask_frame * inverse_crop_frame[:, :, 2] + (1 - inverse_mask_frame) * temp_frame[:, :, 2]
	# 返回粘贴后的帧
	return paste_frame


# 创建一个静态掩码帧，参数为掩码大小、脸部掩码模糊度和脸部掩码填充，返回值为帧
@lru_cache(maxsize = 100)
def create_static_mask_frame(mask_size : Size, face_mask_blur : float, face_mask_padding : Padding) -> Frame:
	# 创建一个掩码帧，大小为mask_size，数据类型为float32
	mask_frame = numpy.ones(mask_size, numpy.float32)
	# 计算模糊量
	blur_amount = int(mask_size[0] * 0.5 * face_mask_blur)
	# 计算模糊区域
	blur_area = max(blur_amount // 2, 1)
	# 设置掩码帧的前后 blur_area 像素为 0
	mask_frame[:max(blur_area, int(mask_size[1] * face_mask_padding[0] / 100)), :] = 0
	mask_frame[-max(blur_area, int(mask_size[1] * face_mask_padding[2] / 100)):, :] = 0
	mask_frame[:, :max(blur_area, int(mask_size[0] * face_mask_padding[3] / 100))] = 0
	mask_frame[:, -max(blur_area, int(mask_size[0] * face_mask_padding[1] / 100)):] = 0
	# 如果模糊量大于0，则使用高斯模糊
	if blur_amount > 0:
		mask_frame = cv2.GaussianBlur(mask_frame, (0, 0), blur_amount * 0.25)
	# 返回掩码帧
	return mask_frame


# 创建静态锚点，使用缓存
@lru_cache(maxsize = 100)
def create_static_anchors(feature_stride : int, anchor_total : int, stride_height : int, stride_width : int) -> numpy.ndarray[Any, Any]:
	# 创建网格，每个网格的行数为stride_height，列数为stride_width
	y, x = numpy.mgrid[:stride_height, :stride_width][::-1]
	# 将y和x合并，得到每个网格的坐标
	anchors = numpy.stack((y, x), axis = -1)
	# 将每个网格的坐标乘以feature_stride，得到每个锚点的坐标
	anchors = (anchors * feature_stride).reshape((-1, 2))
	# 将每个锚点的坐标复制anchor_total次，得到每个锚点的坐标
	anchors = numpy.stack([ anchors ] * anchor_total, axis = 1).reshape((-1, 2))
	# 返回每个锚点的坐标
	return anchors


# 定义一个函数，计算点与边界框之间的距离
# 参数：points：numpy数组，distance：numpy数组
# 返回：边界框
def distance_to_bbox(points : numpy.ndarray[Any, Any], distance : numpy.ndarray[Any, Any]) -> Bbox:
	# 计算x1
	x1 = points[:, 0] - distance[:, 0]
	# 计算y1
	y1 = points[:, 1] - distance[:, 1]
	# 计算x2
	x2 = points[:, 0] + distance[:, 2]
	# 计算y2
	y2 = points[:, 1] + distance[:, 3]
	# 将x1，y1，x2，y2列堆叠
	bbox = numpy.column_stack([ x1, y1, x2, y2 ])
	# 返回边界框
	return bbox


def distance_to_kps(points : numpy.ndarray[Any, Any], distance : numpy.ndarray[Any, Any]) -> Kps:
	x = points[:, 0::2] + distance[:, 0::2]
	y = points[:, 1::2] + distance[:, 1::2]
	kps = numpy.stack((x, y), axis = -1)
	return kps


# 定义函数apply_nms，用于对输入的bbox_list进行非极大值抑制，参数bbox_list为Bbox类型的列表，iou_threshold为抑制的阈值
def apply_nms(bbox_list : List[Bbox], iou_threshold : float) -> List[int]:
	# 定义一个空列表，用于存放保留的索引
	keep_indices = []
	# 将bbox_list转换为numpy数组，并获取每个bbox的宽、高和面积
	dimension_list = numpy.reshape(bbox_list, (-1, 4))
	x1 = dimension_list[:, 0]
	y1 = dimension_list[:, 1]
	x2 = dimension_list[:, 2]
	y2 = dimension_list[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	# 获取bbox_list的索引
	indices = numpy.arange(len(bbox_list))
	# 当索引列表不为空时，循环执行
	while indices.size > 0:
		# 获取第一个索引
		index = indices[0]
		# 获取剩余的索引
		remain_indices = indices[1:]
		# 将第一个索引添加到保留的索引列表中
		keep_indices.append(index)
		# 计算两个bbox的交集的宽、高
		xx1 = numpy.maximum(x1[index], x1[remain_indices])
		yy1 = numpy.maximum(y1[index], y1[remain_indices])
		xx2 = numpy.minimum(x2[index], x2[remain_indices])
		yy2 = numpy.minimum(y2[index], y2[remain_indices])
		# 计算两个bbox的交集的宽度、高度
		width = numpy.maximum(0, xx2 - xx1 + 1)
		height = numpy.maximum(0, yy2 - yy1 + 1)
		# 计算两个bbox的交集的面积
		iou = width * height / (areas[index] + areas[remain_indices] - width * height)
		# 获取iou小于阈值的索引
		indices = indices[numpy.where(iou <= iou_threshold)[0] + 1]
	# 返回保留的索引列表
	return keep_indices
