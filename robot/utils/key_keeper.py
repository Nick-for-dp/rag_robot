import os

from dotenv import load_dotenv


def get_deepseek_api_key(key_name: str = 'DEEPSEEK_API_KEY'):
    """
    从.env文件中读取DEEPSEEK_API_KEY
    
    返回:
        str: DEEPSEEK_API_KEY的值, 如果不存在则返回None
    """
    # 加载.env文件
    load_dotenv()
    
    # 获取环境变量
    api_key = os.environ.get(key_name)
    
    return api_key