import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import json
from datetime import datetime
import pytz
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
# 将父文件夹路径添加到 sys.path
path_ = "IArtPythonCommon"
path = os.path.join(parent_dir,path_)
sys.path.append(path)
sys.path.append(parent_dir)

from IArtPythonCommon.ArtFunc import ArtFunc
from IArtPythonCommon.ArtS3 import ArtUploadDataFromS3
from IArtPythonCommon.ArtEnvironmentVariables import ArtEnvironmentVariables

artFunc = ArtFunc()
artUploadDataFromS3 = ArtUploadDataFromS3()
artEnvironmentVariables = ArtEnvironmentVariables()


class ArtS3:
    def __init__(self, bucket_name = "iartai-luna") -> None:
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.shanghai_tz = pytz.timezone('Asia/Shanghai') # 设置时区为上海

    def _get_data(self):
        # 获取当前时间，并转换为上海时间
        shanghai_time = datetime.now(self.shanghai_tz)
        return shanghai_time.strftime("%Y-%m-%d")

    def download_file_from_s3Key(self, s3_file_key, local_file_path, max_retries=3):
        download_file_successful = False
        for attempt in range(max_retries):
            try:
                self.s3_client.download_file(self.bucket_name, s3_file_key, local_file_path)
                download_file_successful = True
                break
            except Exception as e:
                print(f"下载失败 {e}，正在重试...(尝试次数 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise e  # 达到最大重试次数，抛出异常
        return download_file_successful

    def upload_file_to_s3(self, file_path, max_retries=3):
        uuidName = artFunc.getUuidName("mp4")
        data = self._get_data()

        s3_file_key = f"{artEnvironmentVariables.envName}/generateVideo/{data}/{uuidName}"
        for attempt in range(max_retries):
            try:
                self.s3_client.upload_file(file_path, self.bucket_name, s3_file_key)
                break
            except (BotoCoreError, NoCredentialsError) as e:
                print(f"上传失败，正在重试...(尝试次数 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise e  # 达到最大重试次数，抛出异常
        return s3_file_key
