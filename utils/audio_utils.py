# utils/audio_utils.py
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import warnings

def has_audio_stream(video_path):
    """检查视频文件是否包含音频流"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', 
               '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 'audio' in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn(f"ffprobe not found or error checking audio in {video_path}")
        return False

def extract_audio(video_path, audio_output_path=None):
    """从视频中提取音频"""
    if audio_output_path is None:
        temp_dir = tempfile.mkdtemp()
        audio_output_path = os.path.join(temp_dir, 'extracted_audio.aac')
    
    try:
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'copy', 
               '-y', audio_output_path]
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_output_path, True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"Failed to extract audio: {e}")
        return None, False

def merge_audio_video(video_path, audio_path, output_path):
    """合并视频和音频"""
    try:
        cmd = ['ffmpeg', '-i', video_path, '-i', audio_path,
               '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
               '-shortest', '-y', output_path]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"Failed to merge audio: {e}")
        # 如果合并失败，返回原始视频
        shutil.copy2(video_path, output_path)
        return False

def copy_video_with_audio(input_video_path, output_video_path):
    """
    智能处理音频：如果输入视频有音频，提取并合并到输出视频
    """
    if not has_audio_stream(input_video_path):
        # 没有音频，直接复制视频
        if output_video_path != input_video_path:
            shutil.copy2(input_video_path, output_video_path)
        return True
    
    # 提取音频
    audio_path, success = extract_audio(input_video_path)
    if not success:
        warnings.warn("Failed to extract audio, outputting silent video")
        if output_video_path != input_video_path:
            shutil.copy2(input_video_path, output_video_path)
        return False
    
    try:
        # 创建临时输出路径
        temp_output = output_video_path + '.temp.mp4'
        
        # 合并音频
        merge_success = merge_audio_video(input_video_path, audio_path, temp_output)
        
        if merge_success:
            # 替换原始文件
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(temp_output, output_video_path)
        
        # 清理临时音频文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
            temp_dir = os.path.dirname(audio_path)
            if temp_dir.startswith(tempfile.gettempdir()) and os.path.isdir(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
        
        return merge_success
    except Exception as e:
        warnings.warn(f"Error in audio processing: {e}")
        # 回退：复制无声视频
        if output_video_path != input_video_path:
            shutil.copy2(input_video_path, output_video_path)
        return False