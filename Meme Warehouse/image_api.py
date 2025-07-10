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

# Constant definitions
MAX_RETRIES = 3
TIMEOUT = 1200
API_HOSTNAME = "api2.aigcbest.top"
API_PATH = "/v1/chat/completions"

# Color output constants
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
    """Print colored text"""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.ENDC}")

def print_progress_bar(current: int, total: int, width: int = 50) -> None:
    """Print progress bar"""
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percentage = progress * 100
    print(f"\r{Colors.OKCYAN}Progress: {Colors.ENDC}[{Colors.OKGREEN}{bar}{Colors.ENDC}] {percentage:.1f}% ({current}/{total})", end="", flush=True)

def format_time(seconds: float) -> str:
    """Format time display"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minutes {secs:.1f} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hours {minutes} minutes {secs:.1f} seconds"

#TODO: call the api here
class ImageAPIClient:
    """API client supporting sending text and images simultaneously (no Markdown format required)."""

    def __init__(self, api_key: str, model: str = "gpt-4") -> None:
        self.api_key = api_key
        self.model = model

    # ----------------------- Internal utility methods -----------------------
    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode local image as base64 Data URL."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            mime_type = ImageAPIClient._get_mime_type(image_path)
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            raise RuntimeError(f"Failed to read image {image_path}: {e}")

    @staticmethod
    def _get_mime_type(file_path: str) -> str:
        """Return MIME type based on file extension."""
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

    # ----------------------- Public interface -----------------------
    def call_api(self, text: str, image_paths: Sequence[str] | None = None) -> str:
        """Send text and optional list of images to the model, return model reply."""
        # Build content array
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
                    print_colored(f"❌ Error encoding/adding image {img}: {e}", Colors.FAIL)

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

    # ----------------------- Private methods -----------------------
    def _call_api(self, payload: Dict[str, Any]) -> str:
        """Underlying HTTP call with retry logic."""
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
                    print_colored(
                        f"⚠️  API request failed, status code: {res.status} {res.reason}, error message: {error_data}",
                        Colors.WARNING
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
                print_colored(f"🔄 Network or connection error (attempt {retry}/{MAX_RETRIES}): {e}", Colors.WARNING)
                if retry >= MAX_RETRIES:
                    response_content = f"Error: Connection failed after maximum retries. Last error: {e}"
                    break
                wait_sec = 5 * retry
                print_colored(f"⏳ Waiting {wait_sec} seconds before retrying…", Colors.OKCYAN)
                time.sleep(wait_sec)
            except json.JSONDecodeError as e:
                response_content = (
                    f"Error: Failed to parse API response as JSON. Error: {e}, "
                    f"Response content: {data.decode('utf-8') if 'data' in locals() else 'N/A'}"
                )
                break
            except Exception as e:
                print_colored(f"❌ Unknown error occurred: {e}", Colors.FAIL)
                response_content = f"Error: Unknown error occurred. {e}"
                break
            finally:
                if "conn" in locals() and conn:
                    conn.close()

        return response_content or "Failed to get model response"


# ----------------------- Helper Functions -----------------------

def process_text_and_images(
    text: str,
    image_paths: Sequence[str] | None,
    api_key: str,
    model: str = "gpt-4",
) -> str:
    """One-time call: send text and images."""
    client = ImageAPIClient(api_key=api_key, model=model)
    return client.call_api(text, image_paths)


if __name__ == "__main__":
    # Start time record
    start_time = time.time()
    
    # Beautified title
    print_colored("=" * 80, Colors.HEADER, bold=True)
    print_colored("🎭 Meme Semantic Analysis System", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.HEADER, bold=True)
    print()
    
    # Example: Replace with real API Key 
    example_key = 'sk-Su9jdsVIqpzitVDssWezVOUSqZ8Vqc8MX9ez272iTFnMjYqc'
    content = """你是资深网络语言学专家，请从网络语言角度出发，
分析这张表情包图片适合使用场景、绝对不能使用场景、表情隐含情绪和网络语义、用户发送该表情的心理动机与交流目的，
严格按照以下格式输出，不要包含解释：
"适合使用的场景" :{一句话简洁描述}
"绝对不能使用的场景": {一句话简洁描述}
"表情的隐含情绪和网络语义": {一句话简洁描述}
"用户发送该表情的心理动机与交流目的": {一句话简洁描述}
请确保在回复中保留所有大括号{}，不要删除或替换它们。
其中绝对不能使用的场景不能笼统地说是严肃正式的场合"""

    # Get images
    image_dir = "addtional"
    print_colored(f"📁 Scanning image directory: {image_dir}", Colors.OKBLUE)
    
    try:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))[:]
        if not image_files:
            print_colored("❌ No image files found!", Colors.FAIL)
            exit(1)
        
        print_colored(f"✅ Found {len(image_files)} images", Colors.OKGREEN)
        for i, img in enumerate(image_files, 1):
            print_colored(f"   {i}. {os.path.basename(img)}", Colors.OKCYAN)
        print()
        
    except Exception as e:
        print_colored(f"❌ Error scanning image directory: {e}", Colors.FAIL)
        exit(1)
    
    successful_count = 0
    failed_count = 0
    output_file = "meme-GPT4.json"
    
    # Ensure output file exists and contains a valid JSON array
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    else:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                successful_count = len(existing_results)
                print_colored(f"📖 {successful_count} records found in existing result file", Colors.OKCYAN)
        except Exception as e:
            print_colored(f"⚠️  Existing result file may be corrupted: {e}", Colors.WARNING)
            # Backup possibly corrupted file and create new file
            backup_name = f"{output_file}.bak.{int(time.time())}"
            os.rename(output_file, backup_name)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            successful_count = 0
    
    print_colored("🚀 Start processing images...", Colors.OKBLUE, bold=True)
    print()
    
    for i, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        
        # Check if this image has already been processed
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                current_results = json.load(f)
                already_processed = any(result.get("figure") == img_name for result in current_results)
            if already_processed:
                print_colored(f"⏭️  [{i}/{len(image_files)}] Skipping already processed: {img_name}", Colors.WARNING)
                continue
        except Exception as e:
            print_colored(f"⚠️  Error checking processed images: {e}", Colors.WARNING)
            continue
        
        # Show the image being processed
        print_colored(f"📸 [{i}/{len(image_files)}] Processing: {img_name}", Colors.OKBLUE, bold=True)
        
        # Show progress bar
        print_progress_bar(i-1, len(image_files))
        print()  # New line
        
        img_start_time = time.time()
        
        try:
            print_colored("   🔄 Calling API...", Colors.OKCYAN)
            response = process_text_and_images(content, [img_path], example_key, "gpt-4")
            
            img_end_time = time.time()
            img_duration = img_end_time - img_start_time
            
            print_colored(f"   ✅ API call successful (time taken: {format_time(img_duration)})", Colors.OKGREEN)
            
            # Parse response content
            print_colored("   🔍 Parsing response...", Colors.OKCYAN)
            
            # Extract content within curly braces
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
            
            # Read and update JSON file
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    current_results = json.load(f)
                current_results.append(result)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=4)
                successful_count += 1
                print_colored("   ✅ Parsing complete and saved", Colors.OKGREEN)
            except Exception as save_e:
                print_colored(f"   ❌ Failed to save result: {save_e}", Colors.FAIL)
            
            # Show result summary
            print_colored("   📋 Result summary:", Colors.OKCYAN)
            print_colored(f"      Suitable scenario: {scenarios[:50]}{'...' if len(scenarios) > 50 else ''}", Colors.ENDC)
            
        except Exception as e:
            failed_count += 1
            print_colored(f"   ❌ Processing failed: {e}", Colors.FAIL)
            if 'response' in locals():
                print_colored(f"   🔍 Raw response: {response[:100]}{'...' if len(response) > 100 else ''}", Colors.WARNING)
        
        print()  # Blank line for separation
        
        # Add delay to avoid API rate limit
        if i < len(image_files):  # Not the last image
            print_colored("   ⏸️  Waiting 2 seconds to avoid API limit...", Colors.OKCYAN)
            time.sleep(2)
            print()
    
    # Final progress bar
    print_progress_bar(len(image_files), len(image_files))
    print("\n")
    
    # Statistics
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Get final successful count
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            final_results = json.load(f)
            successful_count = len(final_results)
    except Exception as e:
        print_colored(f"⚠️  Error reading final results: {e}", Colors.WARNING)
    
    print()
    print_colored("=" * 80, Colors.HEADER)
    print_colored("📊 Processing Statistics", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.HEADER)
    print_colored(f"🎯 Total number of images: {len(image_files)}", Colors.OKBLUE)
    print_colored(f"✅ Successfully processed: {successful_count}", Colors.OKGREEN)
    print_colored(f"❌ Failed to process: {failed_count}", Colors.FAIL if failed_count > 0 else Colors.OKGREEN)
    print_colored(f"⏱️  Total time: {format_time(total_duration)}", Colors.OKBLUE)
    print_colored(f"⚡ Average time: {format_time(total_duration/len(image_files))}/image", Colors.OKBLUE)
    print_colored(f"📅 Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.OKBLUE)
    print_colored("=" * 80, Colors.HEADER)
    
    if successful_count > 0:
        print_colored("🎉 Processing complete!", Colors.OKGREEN, bold=True)
    else:
        print_colored("⚠️  All image processing failed, please check error messages", Colors.WARNING, bold=True) 