import os
from fastapi import FastAPI, File, UploadFile
import shutil
from pydantic import BaseModel
import subprocess
from datetime import datetime
import asyncio
import aiohttp
from dotenv import load_dotenv
from typing import Any
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import json
import logging
import os
import shutil
import select
import subprocess
import re

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

logger = logging.getLogger("video")

load_dotenv()

app = FastAPI()

REGION = os.getenv('Region')
BUCKET = os.getenv('Bucket')
PRIVATE_BUCKET = os.getenv('PrivateBucket')
# 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
token = None
scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

config = CosConfig(Region=REGION, SecretId=os.getenv(
    'SecretId'), SecretKey=os.getenv('SecretKey'), Token=token, Scheme=scheme)
cos_client = CosS3Client(config)

async def common_request_callback(url, config):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=config) as response:
            if response.headers.get('Content-Type') == 'application/json':
                return await response.json()
            else:
                return await response.text()

def download_and_extract_list_data(cos_client, PRIVATE_BUCKET, user_id, data_id, model_id, TRAIN_DIR):
    # 下载 JSON 文件
    response = cos_client.get_object(Bucket=PRIVATE_BUCKET, Key=f"user-train/{user_id}/{data_id}/config.json")
    response_body = response["Body"].read()

    # 解析 JSON 数据
    json_data = json.loads(response_body)

    # 提取列表数据
    list_data = json_data["list"]

    # 遍历 list_data 中的图片路径
    for i, image_name in enumerate(list_data):
        # 构建图片路径
        image_path = f"user-train/{user_id}/{data_id}/{image_name}"

        # 下载文件
        # response = cos_client.get_object(Bucket=PRIVATE_BUCKET, Key=image_path)

        # 提取文件名
        file_name = os.path.basename(image_name)
        
        # 拼接本地文件路径
        local_file_path = os.path.join(TRAIN_DIR, file_name)
        
        response = cos_client.get_object(
            Bucket=PRIVATE_BUCKET,
            Key=image_path
        )
        response['Body'].get_stream_to_file(local_file_path)

        print(f"Downloaded file: {local_file_path}")

        # 取第一张图片上传到腾讯云
        if i == 0:
            print("上传封面开始")
            # 构建图片路径
            image_path = f"user-train/{model_id}/cover.jpg"
            cos_client.upload_file(
                Bucket=BUCKET,
                Key=image_path,
                LocalFilePath=local_file_path,
                EnableMD5=False
            )
            print("上传封面成功")


    # 取第一张图片上传到腾讯云
    # 构建图片路径
    # image_path = f"user-train/{user_id}/{data_id}/{list_data[0]}"
    # 下载文件

def upload_files_to_cos(TRAIN_DIR, model_id):
    # 读取目录下的文件并获取前四个txt文件的内容
    file_list = os.listdir(TRAIN_DIR)
    file_list.sort()
    contents = []
    for file in file_list:
        if file.endswith(".txt"):
            with open(os.path.join(TRAIN_DIR, file), 'r', encoding='utf-8') as f:
                contents.append(f.read())
        if len(contents) == 4:
            break

    # 将获取的内容保存为JSON文件
    # 将获取的内容转为JSON格式
    json_content = json.dumps(contents).encode('utf-8')

    # 上传文件到腾讯云
    cos_client.put_object(
        Bucket=BUCKET,
        Key=f"user-train/{model_id}/recommend.json",
        Body=json_content,
        EnableMD5=False
    )

async def run_command(env, callback_url, model_id, server_config, JobURL, *args):
    # Create the subprocess
    process = await asyncio.create_subprocess_exec(
        *args,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    # Compile a regex to match the progress
    progress_regex = re.compile(r"epoch (\d+)/(\d+)")

    # Read output from the subprocess
    async for line in process.stdout:
        line = line.decode().strip()
        print(f'STDOUT: {line}')
        
        # Match the progress
        match = progress_regex.search(line)
        if match:
            current, total = match.groups()
            progress = int(current) / int(total)
            print(f"Progress: {progress*100}%")
            await callback_sync_url(model_id, callback_url, {
                "status": 0,
                "msg": "进行中",
                "process": progress * 100,
                "status": 0,
                "server_config": server_config,
                "model_id": model_id,
                "JobURL": JobURL
            }, 3)

    # Read error from the subprocess
    # async for line in process.stderr:
    #     print(f'STDERR: {line.decode().strip()}')

    # Wait for the subprocess to finish
    await process.wait()

async def callback_sync_url(id, callback_url, config, retry=3):
    print(id, "callbackSync", callback_url, config, retry)
    if retry == 1:
        return

    try:
        print(id, "开始请求=====")
        json_str = await common_request_callback(callback_url, config)
        result = json.loads(json_str)
        print(id, "result===", result)
        if result.get("code") == 0:
            # Success
            print(id, "请求成功")
            pass
        else:
            # Failure, retry up to 3 times with delay
            await asyncio.sleep((4 + 4 - retry) * 1)
            await callback_sync_url(id, callback_url, config, retry - 1)
    except Exception as e:
        print(id, "Error, retrying:", e)
        # Failure, retry up to 3 times with delay
        await asyncio.sleep((4 + 4 - retry) * 1)
        await callback_sync_url(id, callback_url, config, retry - 1)



class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # 获取ID值
        request_id = self.extra['request_id']
        # 在日志消息前添加ID值
        msg = f"[{request_id}] {msg}"
        return msg, kwargs


async def start_jobs(server_config, lr, unet_lr, text_encoder_lr, max_train_epoches, network_dim, network_alpha, type, trigger_word, JobURL, model, model_id, length, data_id, user_id, modelId, callback_url):

    start_time = datetime.now().timestamp()

    logger_adapter = RequestLoggerAdapter(logger, {'request_id': model_id})

    logger_adapter.info(f"参数server_config=={server_config}")
    logger_adapter.info(f"参数lr=={lr}")
    logger_adapter.info(f"参数unet_lr=={unet_lr}")
    logger_adapter.info(f"参数text_encoder_lr=={text_encoder_lr}")
    logger_adapter.info(f"参数max_train_epoches=={max_train_epoches}")
    logger_adapter.info(f"参数network_dim=={network_dim}")
    logger_adapter.info(f"参数network_alpha=={network_alpha}")
    logger_adapter.info(f"参数type=={type}")
    logger_adapter.info(f"参数trigger_word=={trigger_word}")
    logger_adapter.info(f"参数JobURL=={JobURL}")
    logger_adapter.info(f"参数model=={model}")
    logger_adapter.info(f"参数model_id=={model_id}")
    logger_adapter.info(f"参数length=={length}")
    logger_adapter.info(f"参数data_id=={data_id}")
    logger_adapter.info(f"参数user_id=={user_id}")
    logger_adapter.info(f"参数callback_url=={callback_url}")

    try:
        # 参数 
        LORA_NAME = model_id
        KEY_WORD = "" 
        if type == "1":
            KEY_WORD = trigger_word
        elif type == "2":
            KEY_WORD = trigger_word
        elif type == "3":
            KEY_WORD = trigger_word + " style,"
        else:
            KEY_WORD = trigger_word
        ROOT_PATH = "/root/autodl-tmp/lora-scripts"

        TRAIN_DIR = os.path.join(ROOT_PATH, f"{model_id}-input", f"{length}_{model_id}")

        # 创建目录
        os.makedirs(TRAIN_DIR, exist_ok=True)

        # 下载图片
        logger_adapter.info("开始下载图片")
        download_and_extract_list_data(cos_client, PRIVATE_BUCKET, user_id, data_id, model_id, TRAIN_DIR)
        logger_adapter.info("下载图片结束")

        # 执行shell
        logger_adapter.info("开始训练")
        
        env_vars = {
            'ROOT_PATH': ROOT_PATH,
            'FROM_MODEL': model,
            'lr': str(lr),
            'unet_lr': str(unet_lr),
            'text_encoder_lr': str(text_encoder_lr),
            'max_train_epoches': str(max_train_epoches),
            'network_dim': str(network_dim),
            'network_alpha': str(network_alpha),
            'model_ckpt': model,
            'KEY_WORD': KEY_WORD,
            'LORA_NAME': LORA_NAME,
            'TRAIN_DIR': TRAIN_DIR
        }
        try:
            await run_command(env_vars, callback_url, model_id, server_config, JobURL, "bash", "-x", "lora-train.sh")
        except Exception as e:
            # 处理命令执行异常
            logger_adapter.error(f"命令执行失败：{e}")
            return
        
        logger_adapter.info("训练结束")
        
        # 读取文本上传参考词
        logger_adapter.info("开始上传参考词")
        upload_files_to_cos(TRAIN_DIR, model_id)
        logger_adapter.info("上传参考词结束")

        # 上传lora
        logger_adapter.info("开始上传lora")
        shutil.copy(f"{ROOT_PATH}/output/{LORA_NAME}.safetensors", "/root/autodl-nas/Lora")
        logger_adapter.info("上传lora结束")

        # 清除数据
        train_dir = f"{ROOT_PATH}/{LORA_NAME}-input"

        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        else:
            logger_adapter.info("目录不存在")

        file_path = f"{ROOT_PATH}/output/{LORA_NAME}.safetensors"

        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            logger_adapter.info("文件不存在")

        # 回调
        end_time = datetime.now().timestamp()
        duration = end_time - start_time
        logger_adapter.info(f"耗时：{duration}")
        await callback_sync_url(model_id, callback_url, {
            "status": 1,
            "msg": "生成成功",
            "duration": duration,
            "server_config": server_config,
            "model_id": model_id,
            "modelId": modelId,
            "JobURL": JobURL
        }, 3)

    except Exception as e:
        logging.error(e, exc_info=True)
        await callback_sync_url(model_id, callback_url, {
            "status": -1,
            "msg": "生成失败",
            "server_config": server_config,
            "model_id": model_id,
            "modelId": modelId,
            "JobURL": JobURL
        }, 3)


class VideoData(BaseModel):
    server_config: Any
    lr: str
    unet_lr: str
    text_encoder_lr: str
    max_train_epoches: int
    network_dim: int
    network_alpha: int
    type: str
    trigger_word: str
    JobURL: str
    model: str
    data_id: str
    model_id: str
    length: int
    user_id: str
    modelId: str
    callbackUrl: str
    


@app.post("/api/train/start")
async def generate_prompt(data: VideoData):
    asyncio.create_task(start_jobs(data.server_config, data.lr, data.unet_lr, data.text_encoder_lr, data.max_train_epoches, data.network_dim, data.network_alpha, data.type, data.trigger_word, data.JobURL, data.model, data.model_id, data.length, data.data_id, data.user_id, data.modelId, data.callbackUrl))

    return {"code": 0, "msg": "success"}
