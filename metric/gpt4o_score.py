import base64
import json
import os
import time
import http.client
import socket
import ssl
from typing import List, Dict, Any, Sequence
import glob
import re
import argparse

MAX_RETRIES = 3
TIMEOUT = 1200
API_HOSTNAME = "api2.aigcbest.top"
API_PATH = "/v1/chat/completions"

# TODO: Call the API here
class ImageAPIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.api_key = api_key
        self.model = model

    @staticmethod
    def encode_image(image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            mime_type = ImageAPIClient._get_mime_type(image_path)
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            raise RuntimeError(f"Failed to read image {image_path}: {e}")

    @staticmethod
    def _get_mime_type(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".svg": "image/svg+xml",
        }
        return mapping.get(ext, "application/octet-stream")

    def call_api(self, text: str, image_paths: Sequence[str] | None = None) -> str:
        content: List[Dict[str, Any]] = []
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        if image_paths:
            for img in image_paths:
                try:
                    data_url = self.encode_image(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                except Exception as e:
                    print(f"Error encoding/adding image {img}: {e}")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0, # temperature set to 0
            "stream": False  # If the API supports the stream parameter, it is recommended to add it
        }
        return self._call_api(payload)

    def _call_api(self, payload: Dict[str, Any]) -> str:
        payload_str = json.dumps(payload)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        retry = 0
        response_content = None
        while retry < MAX_RETRIES:
            try:
                conn = http.client.HTTPSConnection(API_HOSTNAME, timeout=TIMEOUT)
                conn.request("POST", API_PATH, payload_str, headers)
                res = conn.getresponse()
                if res.status == 200:
                    data = res.read()
                    response_json = json.loads(data.decode("utf-8"))
                    if (
                        response_json.get("choices")
                        and response_json["choices"][0].get("message")
                    ):
                        response_content = (
                            response_json["choices"][0]["message"].get("content")
                        )
                    else:
                        response_content = (
                            "Error: Expected 'content' not found in response. Response: "
                            f"{response_json}"
                        )
                    break
                else:
                    error_data = res.read().decode("utf-8")
                    print(
                        f"API request failed, status code: {res.status} {res.reason}, error info: {error_data}"
                    )
                    if 500 <= res.status < 600:
                        raise Exception(
                            f"Server error: {res.status} {res.reason}, details: {error_data}"
                        )
                    else:
                        response_content = (
                            f"API error: {res.status} {res.reason}, details: {error_data}"
                        )
                        break
            except (
                socket.timeout,
                ssl.SSLError,
                ConnectionResetError,
                ConnectionRefusedError,
                http.client.RemoteDisconnected,
                http.client.NotConnected,
                http.client.CannotSendRequest,
                http.client.ResponseNotReady,
            ) as e:
                retry += 1
                print(f"Network or connection error (attempt {retry}/{MAX_RETRIES}): {e}")
                if retry >= MAX_RETRIES:
                    response_content = f"Error: Connection failed after maximum retries. Last error: {e}"
                    break
                wait_sec = 5 * retry
                print(f"Waiting {wait_sec} seconds before retrying…")
                time.sleep(wait_sec)
            except json.JSONDecodeError as e:
                response_content = (
                    f"Error: Failed to parse API response as JSON. Error: {e}, "
                    f"Response content: {data.decode('utf-8') if 'data' in locals() else 'N/A'}"
                )
                break
            except Exception as e:
                print(f"Unknown error occurred: {e}")
                response_content = f"Error: Unknown error occurred. {e}"
                break
            finally:
                if "conn" in locals() and conn:
                    conn.close()
        return response_content or "Failed to get model response"

def process_text_and_images(
    text: str,
    image_paths: Sequence[str] | None,
    api_key: str,
    model: str = "gpt-4o",
) -> str:
    client = ImageAPIClient(api_key=api_key, model=model)
    return client.call_api(text, image_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch evaluation of meme scores')
    parser.add_argument('--type', type=str, default='role_based_6_turns_Diversity-awareSelection',
                         help='Data type prefix, e.g. role_based_6_turns_Diversity-awareSelection, ' \
                         'role_based_6_turns_GreedySelection,role_based_6_turns_Random, etc.')
    args = parser.parse_args()
    TYPE = args.type
    example_key = 'sk-g7XAqzgl5MxWW0Ln9U94Nm7eZDIIGO6cY4b7k3jLXt1yN1ss'
    content = '''你是网络表情包使用的评委,
    请你结合对话场景,从
    1.聊天语境和表情包表达的语义是否一致
    2.表情包的情绪是否与聊天语境的情绪一致
    3.在对话语境中，发这个表情"合适不尴尬"
    4.表情是否出其不意、幽默反转、恰到好处
    5.这个表情在上下文中是否破坏对话逻辑流，例如突然打断、转移话题等
    五个维度进行分析,针对表情包的使用进行综合打分,分值在0-100,保留1位小数
    严格按照以下格式输出:
    "综合得分":<70.0>
    "原因"<原因>
    输出中保留<>符号,只输出这两项,"原因"中给出综合原因'''
    img_dir = '../Examples/' + TYPE
    output_json = TYPE

    # Resume: Read existing results and collect processed file names
    processed_files = set()
    existing_results = []
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                for item in existing_results:
                    if 'file' in item:
                        processed_files.add(item['file'])
            except Exception:
                pass
    first = len(existing_results) == 0
    # Open in append mode, write existing content first
    with open(output_json, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for idx, item in enumerate(existing_results):
            if idx > 0:
                f.write(',\n')
            f.write(json.dumps(item, ensure_ascii=False, indent=2))
    for fname in sorted(os.listdir(img_dir)):
        if fname.lower().endswith('.png') and fname not in processed_files:
            print(f"Processing {fname}...")
            img_path = os.path.join(img_dir, fname)
            response = process_text_and_images(content, [img_path], example_key, "chatgpt-4o-latest")
            reason = None
            score = None
            print(response)
            if response:
                reason_match = re.search(r'"原因"<([^>]*)>', response)
                score_match = re.search(r'"综合得分":<([0-9.]+)>', response)
                if reason_match:
                    reason = reason_match.group(1)
                if score_match:
                    score = float(score_match.group(1))
            result = {
                'file': fname,
                '原因': reason,
                '综合得分': score
            }
            with open(output_json, 'a', encoding='utf-8') as f:
                if not first:
                    f.write(',\n')
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
                first = False
    with open(output_json, 'a', encoding='utf-8') as f:
        f.write('\n]')
    print(f'Output to {output_json}')