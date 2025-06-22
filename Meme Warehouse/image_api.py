import base64
import json
import os
import time
import http.client
import socket
import ssl
from typing import List, Dict, Any, Sequence
import glob
from datetime import datetime

# 常量定义
MAX_RETRIES = 3
TIMEOUT = 1200
API_HOSTNAME = "api2.aigcbest.top"
API_PATH = "/v1/chat/completions"

# 颜色输出常量
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text: str, color: str = Colors.ENDC, bold: bool = False) -> None:
    """打印彩色文本"""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.ENDC}")

def print_progress_bar(current: int, total: int, width: int = 50) -> None:
    """打印进度条"""
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percentage = progress * 100
    print(f"\r{Colors.OKCYAN}进度: {Colors.ENDC}[{Colors.OKGREEN}{bar}{Colors.ENDC}] {percentage:.1f}% ({current}/{total})", end="", flush=True)

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.1f}秒"


class ImageAPIClient:
    """支持同时发送文本与图片的 API 客户端（无需 Markdown 格式）。"""

    def __init__(self, api_key: str, model: str = "gpt-4") -> None:
        self.api_key = api_key
        self.model = model

    # ----------------------- 内部工具方法 -----------------------
    @staticmethod
    def encode_image(image_path: str) -> str:
        """将本地图片编码为 base64 Data URL。"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            mime_type = ImageAPIClient._get_mime_type(image_path)
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            raise RuntimeError(f"读取图片 {image_path} 失败: {e}")

    @staticmethod
    def _get_mime_type(file_path: str) -> str:
        """根据扩展名返回 MIME 类型。"""
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

    # ----------------------- 公共接口 -----------------------
    def call_api(self, text: str, image_paths: Sequence[str] | None = None) -> str:
        """发送文本和可选图片列表到模型，返回模型回复。"""
        # 构建 content 数组
        content: List[Dict[str, Any]] = []
        if text.strip():
            content.append({"type": "text", "text": text.strip()})

        if image_paths:
            for img in image_paths:
                try:
                    data_url = self.encode_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                except Exception as e:
                    print_colored(f"❌ 编码/添加图片 {img} 时出错: {e}", Colors.FAIL)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 1,
        }

        return self._call_api(payload)

    # ----------------------- 私有方法 -----------------------
    def _call_api(self, payload: Dict[str, Any]) -> str:
        """底层 HTTP 调用，含重试逻辑。"""
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
                            "错误：响应中未找到预期的'content'。响应: "
                            f"{response_json}"
                        )
                    break
                else:
                    error_data = res.read().decode("utf-8")
                    print_colored(
                        f"⚠️  API请求失败，状态码: {res.status} {res.reason}, 错误信息: {error_data}",
                        Colors.WARNING
                    )
                    if 500 <= res.status < 600:
                        raise Exception(
                            f"服务器错误: {res.status} {res.reason}, 详情: {error_data}"
                        )
                    else:
                        response_content = (
                            f"API错误: {res.status} {res.reason}, 详情: {error_data}"
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
                print_colored(f"🔄 网络或连接错误 (尝试 {retry}/{MAX_RETRIES}): {e}", Colors.WARNING)
                if retry >= MAX_RETRIES:
                    response_content = f"错误：达到最大重试次数后连接失败。最后错误: {e}"
                    break
                wait_sec = 5 * retry
                print_colored(f"⏳ 等待 {wait_sec} 秒后重试…", Colors.OKCYAN)
                time.sleep(wait_sec)
            except json.JSONDecodeError as e:
                response_content = (
                    f"错误：无法解析API响应为JSON。错误: {e}, "
                    f"响应内容: {data.decode('utf-8') if 'data' in locals() else 'N/A'}"
                )
                break
            except Exception as e:
                print_colored(f"❌ 发生未知错误: {e}", Colors.FAIL)
                response_content = f"错误：发生未知错误。{e}"
                break
            finally:
                if "conn" in locals() and conn:
                    conn.close()

        return response_content or "未能获取模型响应"


# ----------------------- 辅助函数 -----------------------

def process_text_and_images(
    text: str,
    image_paths: Sequence[str] | None,
    api_key: str,
    model: str = "gpt-4",
) -> str:
    """一次性调用：发送文本与图片。"""
    client = ImageAPIClient(api_key=api_key, model=model)
    return client.call_api(text, image_paths)


if __name__ == "__main__":
    # 开始时间记录
    start_time = time.time()
    
    # 美化的标题
    print_colored("=" * 80, Colors.HEADER, bold=True)
    print_colored("🎭 表情包语义分析系统", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.HEADER, bold=True)
    print()
    
    # 示例：替换为真实 API Key 和图片路径
    example_key = 'sk-Su9jdsVIqpzitVDssWezVOUSqZ8Vqc8MX9ez272iTFnMjYcq'
    content = """你是资深网络语言学专家，请从网络语言角度出发，
分析这张表情包图片适合使用场景、绝对不能使用场景、表情隐含情绪和网络语义、用户发送该表情的心理动机与交流目的，
严格按照以下格式输出，不要包含解释：
"适合使用的场景" :{一句话简洁描述}
"绝对不能使用的场景": {一句话简洁描述}
"表情的隐含情绪和网络语义": {一句话简洁描述}
"用户发送该表情的心理动机与交流目的": {一句话简洁描述}
请确保在回复中保留所有大括号{}，不要删除或替换它们。
其中绝对不能使用的场景不能笼统地说是严肃正式的场合"""

    # 获取图片
    image_dir = "addtional"
    print_colored(f"📁 正在扫描图片目录: {image_dir}", Colors.OKBLUE)
    
    try:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))[:]
        if not image_files:
            print_colored("❌ 未找到任何图片文件！", Colors.FAIL)
            exit(1)
        
        print_colored(f"✅ 找到 {len(image_files)} 张图片", Colors.OKGREEN)
        for i, img in enumerate(image_files, 1):
            print_colored(f"   {i}. {os.path.basename(img)}", Colors.OKCYAN)
        print()
        
    except Exception as e:
        print_colored(f"❌ 扫描图片目录时出错: {e}", Colors.FAIL)
        exit(1)
    
    successful_count = 0
    failed_count = 0
    output_file = "meme-GPT4.json"
    
    # 确保输出文件存在并包含有效的JSON数组
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    else:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                successful_count = len(existing_results)
                print_colored(f"📖 已有结果文件中包含 {successful_count} 条记录", Colors.OKCYAN)
        except Exception as e:
            print_colored(f"⚠️  现有结果文件可能损坏: {e}", Colors.WARNING)
            # 备份可能损坏的文件并创建新文件
            backup_name = f"{output_file}.bak.{int(time.time())}"
            os.rename(output_file, backup_name)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            successful_count = 0
    
    print_colored("🚀 开始处理图片...", Colors.OKBLUE, bold=True)
    print()
    
    for i, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        
        # 检查是否已经处理过这张图片
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                current_results = json.load(f)
                already_processed = any(result.get("figure") == img_name for result in current_results)
            if already_processed:
                print_colored(f"⏭️  [{i}/{len(image_files)}] 跳过已处理: {img_name}", Colors.WARNING)
                continue
        except Exception as e:
            print_colored(f"⚠️  检查已处理图片时出错: {e}", Colors.WARNING)
            continue
        
        # 显示当前处理的图片
        print_colored(f"📸 [{i}/{len(image_files)}] 正在处理: {img_name}", Colors.OKBLUE, bold=True)
        
        # 显示进度条
        print_progress_bar(i-1, len(image_files))
        print()  # 换行
        
        img_start_time = time.time()
        
        try:
            print_colored("   🔄 正在调用 API...", Colors.OKCYAN)
            response = process_text_and_images(content, [img_path], example_key, "gpt-4")
            
            img_end_time = time.time()
            img_duration = img_end_time - img_start_time
            
            print_colored(f"   ✅ API 调用成功 (耗时: {format_time(img_duration)})", Colors.OKGREEN)
            
            # 解析响应内容
            print_colored("   🔍 正在解析响应...", Colors.OKCYAN)
            
            # 提取大括号中的内容
            scenarios = response.split("{")[1].split("}")[0].strip()
            inappropriate = response.split("{")[2].split("}")[0].strip()
            emotions = response.split("{")[3].split("}")[0].strip()
            motivation = response.split("{")[4].split("}")[0].strip()
            
            result = {
                "figure": img_name,
                "1": scenarios,
                "2": inappropriate,
                "3": emotions,
                "4": motivation
            }
            
            # 读取并更新JSON文件
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    current_results = json.load(f)
                current_results.append(result)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=4)
                successful_count += 1
                print_colored("   ✅ 解析完成并保存", Colors.OKGREEN)
            except Exception as save_e:
                print_colored(f"   ❌ 保存结果失败: {save_e}", Colors.FAIL)
            
            # 显示解析结果摘要
            print_colored("   📋 结果摘要:", Colors.OKCYAN)
            print_colored(f"      适合场景: {scenarios[:50]}{'...' if len(scenarios) > 50 else ''}", Colors.ENDC)
            
        except Exception as e:
            failed_count += 1
            print_colored(f"   ❌ 处理失败: {e}", Colors.FAIL)
            if 'response' in locals():
                print_colored(f"   🔍 原始响应: {response[:100]}{'...' if len(response) > 100 else ''}", Colors.WARNING)
        
        print()  # 空行分隔
        
        # 添加延时避免API限制
        if i < len(image_files):  # 不是最后一张图片
            print_colored("   ⏸️  等待 2 秒避免 API 限制...", Colors.OKCYAN)
            time.sleep(2)
            print()
    
    # 最终进度条
    print_progress_bar(len(image_files), len(image_files))
    print("\n")
    
    # 统计信息
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 获取最终的处理成功数量
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            final_results = json.load(f)
            successful_count = len(final_results)
    except Exception as e:
        print_colored(f"⚠️  读取最终结果时出错: {e}", Colors.WARNING)
    
    print()
    print_colored("=" * 80, Colors.HEADER)
    print_colored("📊 处理统计", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.HEADER)
    print_colored(f"🎯 总图片数量: {len(image_files)}", Colors.OKBLUE)
    print_colored(f"✅ 处理成功: {successful_count}", Colors.OKGREEN)
    print_colored(f"❌ 处理失败: {failed_count}", Colors.FAIL if failed_count > 0 else Colors.OKGREEN)
    print_colored(f"⏱️  总耗时: {format_time(total_duration)}", Colors.OKBLUE)
    print_colored(f"⚡ 平均耗时: {format_time(total_duration/len(image_files))}/张", Colors.OKBLUE)
    print_colored(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.OKBLUE)
    print_colored("=" * 80, Colors.HEADER)
    
    if successful_count > 0:
        print_colored("🎉 处理完成！", Colors.OKGREEN, bold=True)
    else:
        print_colored("⚠️  所有图片处理都失败了，请检查错误信息", Colors.WARNING, bold=True) 