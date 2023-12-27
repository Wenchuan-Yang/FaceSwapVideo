import os
import time
import json
import sys
# 获取当前文件的绝对路径和文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 将父文件夹路径添加到 sys.path
path_ = "IArtPythonCommon"
path = os.path.join(current_dir,path_)

sys.path.append(parent_dir)
sys.path.append(path)


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
# from torch.multiprocessing import Process, Queue, set_start_method
from IArtPythonCommon.ArtBoto import ArtBoto
from IArtPythonCommon.ArtMetric import ArtMetric
from IArtPythonCommon.ArtSQSTask import ArtSQSTask
from IArtPythonCommon.ArtEnvironmentVariables import ArtEnvironmentVariables
from IArtPythonCommon.ArtGracefulShutDown import ArtGracefulShutDown
warnings.simplefilter('ignore')
# 线程/进程数
thread_count = ArtEnvironmentVariables.threadCount
artBoto = ArtBoto()
artMetric = ArtMetric(60, artBoto)
isLocalTest = os.environ.get('IS_LOCAL_TEST', 'True') == 'True'


def getSqsTaskToQueue():
    """
    en:Get messages from SQS; cn:从SQS获取消息

    返回：
    - 获取的消息
    """
    task_list = None
    try:
        art_taskList = artBoto.sqsGetInferenceExecute()
        if art_taskList != None:
            task_list = art_taskList
        else: # en: No messages were obtained from SQS then we stop emitting processing message signal and instead emit waiting signal. cn: 从SQS没有获取到消息则我们停止发射处理消息信号，而是发射等待信号。
            artMetric.publishWaitingForMessageMetric()
    except Exception as e: 
        logging.warning(f"en: An exception occurred in getting messages from SQS in IArtPythonCommon. cn: IArtPythonCommon 中从SQS获取消息发生异常:", e)
    return task_list
    
def artWorker(mainProcessingLogic):
    """
    en:main processing logic; cn:主要的处理逻辑

    参数：
    - mainProcessingLogic: en:main processing method; cn:主要的处理方法
    - q: en:local message queue; cn:本地的消息队列

    返回：
    - 无
    """
    logging.info("en:The child process starts running; en:子进程开始执行")
    while True:
        try:
            art_task_list = getSqsTaskToQueue()
            if art_task_list is not None and len(art_task_list) > 0:
                for art_task in art_task_list:
                    try: 
                        data_dict = art_task.bodyDict
                        # en:Data processing; cn:处理数据
                        result = mainProcessingLogic(data_dict)
                        # en:Send the result to the message queue; cn:发送结果到消息队列
                        artBoto.sqsSendInferenceCallback(json.dumps(result))

                        # en:Delete this message from SQS; cn:从SQS删除此条消息
                        artBoto.sqsRemoveMessage(message=art_task.task)
                    except Exception as e:
                        logging.warning("en:Message processing exception in IArtPythonCommon; \
                                cn: IArtPythonCommon中消息处理异常: "+ str(e))
                        artMetric.publishErrorMetric()
                        if isinstance(art_task, ArtSQSTask):
                            art_task.incrementRetryCount()
                            if art_task.getRetryCount() > 2:
                                logging.info("en: sending task to DLQ; cn: 发送任务到DLQ")
                                artBoto.sqsSendInferenceExecuteDLQ(art_task.stringifiedBody())
                            else:
                                logging.info("en: sending task to back to queue for retry; cn: 将任务发送回队列以重试")
                                artBoto.sqsSendInferenceExecute(art_task.stringifiedBody())
                            artBoto.sqsRemoveMessage(message=art_task.task)

        except Exception as e:
            logging.warning("en:Exception in artWorker; cn: artWorker 中异常: "+ str(e))
        time.sleep(0.1)

def iartMain(mainFunc):
    """
    en:main method entry; cn:主要的方法入口

    参数：
    - mainFunc: en:A method for processing json data; cn:一个处理json数据方法
    - loadModelTime: en:Time to load the model; cn:加载模型的时间

    返回：
    - None
    """

    # en:Loop listens to the local message queue; cn:循环监听本地消息队列
    while True:
        try:
            artWorker(mainFunc)
            artGracefulShutdown = ArtGracefulShutDown()
            instanceId = artGracefulShutdown.get_instanceId()
            if artGracefulShutdown.gotKillSignal():
                artMetric.publishWaitingForTakeDownMetric()
                instance = f"instanceId:{instanceId} "
                logging.info(instance + "en: Received kill signal will no longer accept messages. cn: 收到 kill 信号，将不再接受消息。")
                break
            else:
                time.sleep(2)
        except Exception as e:
            logging.warning("en:Exception in main process in IArtPythonCommon; cn:IArtPythonCommon 中的主进程发生异常:", e)