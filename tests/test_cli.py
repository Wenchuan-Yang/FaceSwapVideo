import subprocess
import sys
import pytest

import os
import sys

# 获取当前文件的绝对路径和文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from facefusion import wording
from facefusion.utilities import conditional_download


# @pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download('.assets/examples',
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.jpg',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/target-1080p.mp4'
	])
	subprocess.run([ 'ffmpeg', '-i', '.assets/examples/target-1080p.mp4', '-vframes', '1', '.assets/examples/target-1080p.jpg' ])


def test_image_to_image() -> None:
	commands = [ sys.executable, 'run.py', '-s', '.assets/examples/source.jpg', '-t', '.assets/examples/target-1080p.jpg', '-o', '.assets/examples', '--headless' ]
	run = subprocess.run(commands, stdout = subprocess.PIPE)

	assert run.returncode == 0
	assert wording.get('processing_image_succeed') in run.stdout.decode()


def test_image_to_video() -> None:
	commands = [ sys.executable, 'run.py', '-s', '.assets/examples/source.jpg', '-t', '.assets/examples/target-1080p.mp4', '-o', '.assets/examples', '--trim-frame-end', '10', '--headless' ]
	run = subprocess.run(commands, stdout = subprocess.PIPE)

	assert run.returncode == 0
	assert wording.get('processing_video_succeed') in run.stdout.decode()

if __name__ == "__main__":
	# before_all()
	# test_image_to_image()
	test_image_to_video()