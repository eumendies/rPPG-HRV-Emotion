import json
import uuid
from typing import Dict, Tuple, Optional

import numpy as np
import requests


def send_process_data(
        detection_uuid: str,
        student_id: int,
        data_array: list,
        url: str = "http://localhost:8080/air/processdata/add",
        headers: Dict[str, str] = None
) -> Tuple[bool, Dict]:
    """
    发送过程数据存储请求

    Args:
        detection_uuid: 检测UUID
        student_id: 学生ID
        data_array: 三维数组数据
        url: 接口地址
        headers: 自定义请求头（可选）

    Returns:
        Tuple[bool, Dict]: (是否成功, 响应数据)
    """
    try:
        payload = {
            "detectionUUID": detection_uuid,
            "studentID": student_id,
            "dataArray": json.dumps(data_array)  # 三维数组转JSON字符串
        }

        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        response = requests.post(url, json=payload, headers=default_headers)
        return True, response.json()

    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def get_student_info(
        student_id: int,
        base_url: str = "http://localhost:8080",
        headers: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict]:
    """
    学号登录

    Args:
        student_id: 学生ID
        base_url: 基础URL (默认: http://localhost:8080)
        headers: 自定义请求头 (可选)

    Returns:
        Tuple[bool, Dict]: (是否成功, 响应数据)
    """
    try:
        url = f"{base_url}/air/selectStudent/{student_id}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 200:  # 假设AjaxResult的成功状态码为200
                return True, result.get("data", {})
            return False, {"error": result.get("msg", "未知错误")}

        return False, {"error": f"HTTP错误码: {response.status_code}"}

    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def get_process_data_array(
        detection_uuid: str,
        student_id: int,
        base_url: str = "http://localhost:8080",
        headers: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict]:
    """
    获取过程数据存储详细信息

    Args:
        detection_uuid: 检测UUID
        student_id: 学生ID
        base_url: 基础URL (默认: http://localhost:8080)
        headers: 自定义请求头 (可选)

    Returns:
        Tuple[bool, Dict]: (是否成功, 响应数据)
    """
    try:
        url = f"{base_url}/air/processdata/{detection_uuid}/{student_id}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 200:  # 假设AjaxResult的成功状态码为200
                return True, result.get("data", {})
            return False, {"error": result.get("msg", "未知错误")}

        return False, {"error": f"HTTP错误码: {response.status_code}"}

    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


if __name__ == "__main__":
    print(get_student_info(1))

    # 使用循环构造一个形状为[3, 1024, 3]的三维数组
    array_3d = np.zeros((3, 1024, 3))

    # 使用循环填充数组
    for i in range(array_3d.shape[0]):  # 遍历第一个维度（大小为3）
        for j in range(array_3d.shape[1]):  # 遍历第二个维度（大小为1024）
            for k in range(array_3d.shape[2]):  # 遍历第三个维度（大小为3）
                array_3d[i, j, k] = i + j + k  # 示例：将索引的和作为值

    detection_id = uuid.uuid4().hex
    print(send_process_data(detection_id, 1, array_3d.tolist()))
    print(get_process_data_array(detection_id, 1))
