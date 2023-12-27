import os
is_local_test = os.environ.get('IS_LOCAL_TEST', 'True') == 'True'
import time
import sys
import shutil
# 获取当前文件的绝对路径和文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 将父文件夹路径添加到 sys.path
path_ = "IArtPythonCommon"
path = os.path.join(current_dir,path_)
sys.path.append(parent_dir)

if is_local_test:
    from dotenv import load_dotenv
    load_dotenv()  # 加载.env文件中的环境变量

# from scripts import download_weights_from_S3
# # 初始化生成
# time1 = time.time()
# download_weights_from_S3.download_weights()
# print(f"下载模型权重耗时：{time.time() - time1}")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from ArtLunaVideoProcessingLoop import iartMain
from facefusion.core import run
import facefusion.globals
from art_model.art_s3 import ArtS3

art_s3 = ArtS3()


def process_data_main(data):
    time_start = time.time()
    logging.info(f"获取到参数:{data}")

    message_id = data['messageId']
    image_s3_path = data['imagePath']
    video_s3_path = data['videoPath']

    code = 200
    message = "处理成功！"
    result_s3_key = None
    try:
        # 创建临时文件夹
        tmp_path = "tmp"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        # 从S3下载图像
        image_name = os.path.basename(image_s3_path)
        local_image_path = f"{tmp_path}/{image_name}"
        art_s3.download_file_from_s3Key(image_s3_path, local_image_path)
        # 从S3下载视频
        video_name = os.path.basename(video_s3_path)
        local_video_path = f"{tmp_path}/{video_name}"
        art_s3.download_file_from_s3Key(video_s3_path, local_video_path)

        # 对数据进行处理
        # result_name = "result_" + video_name
        # result_path = f"{tmp_path}/{result_name}"
        facefusion.globals.source_path = local_image_path
        facefusion.globals.target_path = local_video_path
        facefusion.globals.output_path = tmp_path
        run()
        result_path = os.path.join(facefusion.globals.output_path, "result.mp4")
        # 上传结果视频到S3
        logging.info(f"结束处理messageId={message_id}, 开始上传结果到S3")
        result_s3_key = art_s3.upload_file_to_s3(result_path)
        
    except Exception as e:
        code = 400
        message = "处理失败！"
        logging.warning(f"处理(messageId:{message_id})过程中发生异常: {e}")

    # # 删除临时文件夹
    # try:
    #     shutil.rmtree(tmp_path)
    # except OSError as e:
    #     logging.warning(f"错误: {tmp_path} : {e.strerror}")

    result = {
        "code": code,
        "message":message,
        "messageId":message_id,
        "resultPath":result_s3_key
    }
    logging.info(f"处理messageId={message_id}结果：{result}")
    logging.info(f"处理messageId={message_id}耗时：{time.time() - time_start}")
    return result

if __name__ == "__main__":
    iartMain(mainFunc=process_data_main)