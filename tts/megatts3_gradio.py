# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import torch
import os
import json
import time
import shutil
import re
import wave
import numpy as np
from pathlib import Path
from functools import partial
import gradio as gr
import traceback
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav

# Default voice library path
DEFAULT_VOICE_LIBRARY = "voice_library"
CONFIG_FILE = "megatts3_config.json"

def load_config():
    """Load configuration including voice library path"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('voice_library_path', DEFAULT_VOICE_LIBRARY)
        except:
            return DEFAULT_VOICE_LIBRARY
    return DEFAULT_VOICE_LIBRARY

def save_config(voice_library_path):
    """Save configuration including voice library path"""
    config = {
        'voice_library_path': voice_library_path,
        'last_updated': str(time.time())
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return f"✅ Configuration saved - Voice library path: {voice_library_path}"
    except Exception as e:
        return f"❌ Error saving configuration: {str(e)}"

def ensure_voice_library_exists(voice_library_path):
    """Ensure the voice library directory exists"""
    Path(voice_library_path).mkdir(parents=True, exist_ok=True)
    return voice_library_path

def get_voice_profiles(voice_library_path):
    """Get list of saved voice profiles"""
    if not os.path.exists(voice_library_path):
        return []
    
    profiles = []
    for item in os.listdir(voice_library_path):
        profile_path = os.path.join(voice_library_path, item)
        if os.path.isdir(profile_path):
            config_file = os.path.join(profile_path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    profiles.append({
                        'name': item,
                        'display_name': config.get('display_name', item),
                        'description': config.get('description', ''),
                        'config': config
                    })
                except:
                    continue
    return profiles

def get_voice_choices(voice_library_path):
    """Get voice choices for dropdown with display names"""
    profiles = get_voice_profiles(voice_library_path)
    choices = [("Manual Input (Upload Audio)", None)]  # Default option
    for profile in profiles:
        display_text = f"🎭 {profile['display_name']} ({profile['name']})"
        choices.append((display_text, profile['name']))
    return choices

def get_audiobook_voice_choices(voice_library_path):
    """Get voice choices for audiobook creation (no manual input option)"""
    profiles = get_voice_profiles(voice_library_path)
    choices = []
    if not profiles:
        choices.append(("No voices available - Create voices first", None))
    else:
        for profile in profiles:
            display_text = f"🎭 {profile['display_name']} ({profile['name']})"
            choices.append((display_text, profile['name']))
    return choices

def save_voice_profile(voice_library_path, voice_name, display_name, description, audio_file, 
                      infer_timestep=32, p_w=1.4, t_w=3.0):
    """Save a voice profile with MegaTTS3 settings"""
    if not voice_name:
        return "❌ Error: Voice name cannot be empty"
    
    # Sanitize voice name for folder
    safe_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')
    
    if not safe_name:
        return "❌ Error: Invalid voice name"
    
    ensure_voice_library_exists(voice_library_path)
    
    profile_dir = os.path.join(voice_library_path, safe_name)
    os.makedirs(profile_dir, exist_ok=True)
    
    # Handle audio file
    audio_path = None
    if audio_file:
        audio_ext = os.path.splitext(audio_file)[1]
        audio_path = os.path.join(profile_dir, f"reference{audio_ext}")
        shutil.copy2(audio_file, audio_path)
        # Store relative path
        audio_path = f"reference{audio_ext}"
    
    # Save configuration with MegaTTS3 parameters
    config = {
        "display_name": display_name or voice_name,
        "description": description or "",
        "audio_file": audio_path,
        "infer_timestep": infer_timestep,
        "p_w": p_w,
        "t_w": t_w,
        "created_date": str(time.time()),
        "version": "1.0"
    }
    
    config_file = os.path.join(profile_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return f"✅ Voice profile '{display_name or voice_name}' saved successfully!"

def load_voice_profile(voice_library_path, voice_name):
    """Load a voice profile and return its settings"""
    if not voice_name:
        return None, 32, 1.4, 3.0, "No voice selected"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if not os.path.exists(config_file):
        return None, 32, 1.4, 3.0, f"❌ Voice profile '{voice_name}' not found"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        audio_file = None
        if config.get('audio_file'):
            audio_path = os.path.join(profile_dir, config['audio_file'])
            if os.path.exists(audio_path):
                audio_file = audio_path
        
        return (
            audio_file,
            config.get('infer_timestep', 32),
            config.get('p_w', 1.4),
            config.get('t_w', 3.0),
            f"✅ Loaded voice profile: {config.get('display_name', voice_name)}"
        )
    except Exception as e:
        return None, 32, 1.4, 3.0, f"❌ Error loading voice profile: {str(e)}"

def delete_voice_profile(voice_library_path, voice_name):
    """Delete a voice profile"""
    if not voice_name:
        return "❌ No voice selected", []
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    if os.path.exists(profile_dir):
        try:
            shutil.rmtree(profile_dir)
            return f"✅ Voice profile '{voice_name}' deleted successfully!", get_voice_profiles(voice_library_path)
        except Exception as e:
            return f"❌ Error deleting voice profile: {str(e)}", get_voice_profiles(voice_library_path)
    else:
        return f"❌ Voice profile '{voice_name}' not found", get_voice_profiles(voice_library_path)

def refresh_voice_list(voice_library_path):
    """Refresh the voice profile list"""
    profiles = get_voice_profiles(voice_library_path)
    choices = [p['name'] for p in profiles]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

def refresh_voice_choices(voice_library_path):
    """Refresh voice choices for TTS dropdown"""
    choices = get_voice_choices(voice_library_path)
    return gr.Dropdown(choices=choices, value=None)

def refresh_audiobook_voice_choices(voice_library_path):
    """Refresh voice choices for audiobook creation"""
    choices = get_audiobook_voice_choices(voice_library_path)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices and choices[0][1] else None)

def load_voice_for_tts(voice_library_path, voice_name):
    """Load a voice profile for TTS tab - returns settings for sliders"""
    if not voice_name:
        # Return to manual input mode
        return None, 32, 1.4, 3.0, gr.Audio(visible=True), "📝 Manual input mode - upload your own audio file below"
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if not os.path.exists(config_file):
        return None, 32, 1.4, 3.0, gr.Audio(visible=True), f"❌ Voice profile '{voice_name}' not found"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        audio_file = None
        if config.get('audio_file'):
            audio_path = os.path.join(profile_dir, config['audio_file'])
            if os.path.exists(audio_path):
                audio_file = audio_path
        
        # Hide manual audio upload when using saved voice
        audio_component = gr.Audio(visible=False) if audio_file else gr.Audio(visible=True)
        
        status_msg = f"✅ Using voice: {config.get('display_name', voice_name)}"
        if config.get('description'):
            status_msg += f" - {config['description']}"
        
        return (
            audio_file,
            config.get('infer_timestep', 32),
            config.get('p_w', 1.4),
            config.get('t_w', 3.0),
            audio_component,
            status_msg
        )
    except Exception as e:
        return None, 32, 1.4, 3.0, gr.Audio(visible=True), f"❌ Error loading voice profile: {str(e)}"

def chunk_text_by_sentences(text, max_words=50):
    """Split text into chunks, breaking at sentence boundaries after reaching max_words"""
    # Split text into sentences using regex to handle multiple punctuation marks
    sentences = re.split(r'([.!?]+\s*)', text)
    
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
            
        # Add punctuation if it exists
        if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words, start new chunk
        if current_word_count > 0 and current_word_count + sentence_words > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_word_count = sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def load_text_file(file_path):
    """Load text from uploaded file"""
    if file_path is None:
        return "No file uploaded", "❌ Please upload a text file"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic validation
        if not content.strip():
            return "", "❌ File is empty"
        
        word_count = len(content.split())
        char_count = len(content)
        
        status = f"✅ File loaded successfully!\n📄 {word_count:,} words | {char_count:,} characters"
        
        return content, status
        
    except UnicodeDecodeError:
        try:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            word_count = len(content.split())
            char_count = len(content)
            status = f"✅ File loaded (latin-1 encoding)!\n📄 {word_count:,} words | {char_count:,} characters"
            return content, status
        except Exception as e:
            return "", f"❌ Error reading file: {str(e)}"
    except Exception as e:
        return "", f"❌ Error loading file: {str(e)}"

def validate_audiobook_input(text_content, selected_voice, project_name):
    """Validate inputs for audiobook creation"""
    issues = []
    
    if not text_content or not text_content.strip():
        issues.append("📝 Text content is required")
    
    if not selected_voice:
        issues.append("🎭 Voice selection is required")
    
    if not project_name or not project_name.strip():
        issues.append("📁 Project name is required")
    
    if text_content and len(text_content.strip()) < 10:
        issues.append("📏 Text is too short (minimum 10 characters)")
    
    if issues:
        return (
            gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=False),
            "❌ Please fix these issues:\n" + "\n".join(f"• {issue}" for issue in issues), 
            gr.Audio(visible=False)
        )
    
    word_count = len(text_content.split())
    chunks = chunk_text_by_sentences(text_content)
    chunk_count = len(chunks)
    
    return (
        gr.Button("🎵 Create Audiobook", variant="primary", size="lg", interactive=True),
        f"✅ Ready for audiobook creation!\n📊 {word_count:,} words → {chunk_count} chunks\n📁 Project: {project_name.strip()}", 
        gr.Audio(visible=True)
    )

def get_voice_config(voice_library_path, voice_name):
    """Get voice configuration for audiobook generation"""
    if not voice_name:
        return None
    
    profile_dir = os.path.join(voice_library_path, voice_name)
    config_file = os.path.join(profile_dir, "config.json")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            audio_file = None
            if config.get('audio_file'):
                audio_path = os.path.join(profile_dir, config['audio_file'])
                if os.path.exists(audio_path):
                    audio_file = audio_path
            
            return {
                'audio_file': audio_file,
                'infer_timestep': config.get('infer_timestep', 32),
                'p_w': config.get('p_w', 1.4),
                't_w': config.get('t_w', 3.0),
                'display_name': config.get('display_name', voice_name)
            }
        except Exception as e:
            print(f"⚠️ Error reading config for voice '{voice_name}': {str(e)}")
            return None
    
    print(f"❌ Voice '{voice_name}' not found")
    return None

def create_audiobook_single(input_queue, output_queue, text_content, voice_library_path, selected_voice, project_name):
    """Create audiobook from text using selected voice with chunking"""
    if not text_content or not selected_voice or not project_name:
        return None, "❌ Missing required fields"

    # Get voice configuration
    voice_config = get_voice_config(voice_library_path, selected_voice)
    if not voice_config:
        return None, f"❌ Could not load voice configuration for '{selected_voice}'"
    if not voice_config['audio_file']:
        return None, f"❌ No audio file found for voice '{voice_config['display_name']}'"

    # Chunk text
    chunks = chunk_text_by_sentences(text_content, max_words=50)
    total_chunks = len(chunks)
    if total_chunks == 0:
        return None, "❌ No text chunks to process"

    # Project directory
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    project_dir = os.path.join("audiobook_projects", safe_project_name)
    os.makedirs(project_dir, exist_ok=True)

    audio_chunks = []
    
    for i, chunk_text in enumerate(chunks, 1):
        try:
            print(f"🎵 Processing chunk {i}/{total_chunks}: {chunk_text[:50]}...")
            
            # Use MegaTTS3 inference
            input_queue.put((
                voice_config['audio_file'], 
                chunk_text, 
                voice_config['infer_timestep'], 
                voice_config['p_w'], 
                voice_config['t_w']
            ))
            
            result = output_queue.get()
            
            if result is None:
                return None, f"❌ Error generating chunk {i}"
            
            # Save individual chunk
            chunk_filename = f"{safe_project_name}_{i:03d}.wav"
            chunk_path = os.path.join(project_dir, chunk_filename)
            
            with open(chunk_path, 'wb') as f:
                f.write(result)
            
            # Load for combining
            with wave.open(chunk_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                audio_chunks.append(audio_data)
            
        except Exception as e:
            return None, f"❌ Error generating chunk {i}: {str(e)}"

    # Combine all audio for preview
    if audio_chunks:
        combined_audio = np.concatenate(audio_chunks)
        sample_rate = 24000  # Default sample rate
        
        total_words = len(text_content.split())
        duration_minutes = len(combined_audio) // sample_rate // 60
        
        success_msg = f"✅ Audiobook created successfully!\n🎭 Voice: {voice_config['display_name']}\n📊 {total_words:,} words in {total_chunks} chunks\n⏱️ Duration: ~{duration_minutes} minutes\n📁 Saved to: {project_dir}"
        
        return (sample_rate, combined_audio), success_msg
    else:
        return None, "❌ No audio generated"

def clean_character_name_from_text(text, voice_name):
    """Remove character name from the beginning of text if it matches the voice name"""
    text = text.strip()
    if not text:
        return ""
    
    # Simple patterns to filter out voice-only lines
    if re.match(r'^\[[^\]]+\]\s*$', text):
        return ""
    
    # If text is just the voice name, return empty
    if text.lower().strip() == voice_name.lower().strip():
        return ""
    
    # Remove voice name from beginning of text
    voice_patterns = [
        voice_name,
        voice_name.upper(),
        voice_name.lower(),
        voice_name.capitalize()
    ]
    
    for voice_var in voice_patterns:
        # Check for "VoiceName: text" or "VoiceName text"
        if text.startswith(voice_var + ":"):
            cleaned = text[len(voice_var + ":"):].strip()
            if cleaned:
                return cleaned
        elif text.startswith(voice_var + " "):
            cleaned = text[len(voice_var + " "):].strip()
            if cleaned:
                return cleaned
    
    return text

def parse_multi_voice_text(text):
    """Parse text with voice tags like [voice_name] and return segments with associated voices"""
    # Split text by voice tags but keep the tags
    pattern = r'(\[([^\]]+)\])'
    parts = re.split(pattern, text)
    
    segments = []
    current_voice = None
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        if not part:
            i += 1
            continue
            
        # Check if this is a voice tag
        if part.startswith('[') and part.endswith(']'):
            # This is a voice tag
            current_voice = part[1:-1]  # Remove brackets
            i += 1
        else:
            # This is text content
            if part and current_voice:
                # Clean the text by removing character name if it matches the voice tag
                cleaned_text = clean_character_name_from_text(part, current_voice)
                # Only add non-empty segments after cleaning
                if cleaned_text.strip():
                    segments.append((current_voice, cleaned_text))
                else:
                    print(f"[DEBUG] Skipping empty segment after cleaning for voice '{current_voice}'")
            elif part:
                # Text without voice tag - use default
                segments.append((None, part))
            i += 1
    
    return segments

def chunk_multi_voice_segments(segments, max_words=50):
    """Take voice segments and chunk them appropriately while preserving voice assignments"""
    final_chunks = []
    
    for voice_name, text in segments:
        # Chunk this segment using the same sentence boundary logic
        text_chunks = chunk_text_by_sentences(text, max_words)
        
        # Add voice assignment to each chunk
        for chunk in text_chunks:
            final_chunks.append((voice_name, chunk))
    
    return final_chunks

def validate_multi_voice_text(text_content, voice_library_path):
    """Validate multi-voice text and check if all referenced voices exist"""
    if not text_content or not text_content.strip():
        return False, "❌ Text content is required", {}
    
    # Parse the text to find voice references
    segments = parse_multi_voice_text(text_content)
    
    if not segments:
        return False, "❌ No valid voice segments found", {}
    
    # Count voice usage and check availability
    voice_counts = {}
    missing_voices = []
    available_voices = [p['name'] for p in get_voice_profiles(voice_library_path)]
    
    for voice_name, text_segment in segments:
        if voice_name is None:
            voice_name = "No Voice Tag"
        
        if voice_name not in voice_counts:
            voice_counts[voice_name] = 0
        voice_counts[voice_name] += len(text_segment.split())
        
        # Check if voice exists (skip None/default)
        if voice_name != "No Voice Tag" and voice_name not in available_voices:
            if voice_name not in missing_voices:
                missing_voices.append(voice_name)
    
    if missing_voices:
        return False, f"❌ Missing voices: {', '.join(missing_voices)}", voice_counts
    
    if "No Voice Tag" in voice_counts:
        return False, "❌ Found text without voice tags. All text must be assigned to a voice using [voice_name]", voice_counts
    
    return True, "✅ All voices found and text properly tagged", voice_counts

def batch_model_worker(input_queue, output_queue, device_id, batch_size=3):
    """Enhanced model worker that processes multiple requests in batches"""
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        # Collect batch of tasks
        batch_tasks = []
        
        # Get first task (blocking)
        try:
            first_task = input_queue.get(timeout=1)  # Wait up to 1 second
            batch_tasks.append(first_task)
        except:
            continue  # Timeout, try again
        
        # Try to get more tasks for batch (non-blocking)
        for _ in range(batch_size - 1):
            try:
                task = input_queue.get_nowait()
                batch_tasks.append(task)
            except:
                break  # No more tasks available
        
        print(f"Processing batch of {len(batch_tasks)} tasks")
        
        # Process batch
        batch_results = []
        for task in batch_tasks:
            inp_audio_path, inp_text, infer_timestep, p_w, t_w = task
            try:
                convert_to_wav(inp_audio_path)
                wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
                cut_wav(wav_path, max_len=28)
                with open(wav_path, 'rb') as file:
                    file_content = file.read()
                resource_context = infer_pipe.preprocess(file_content, latent_file=None)
                wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
                batch_results.append(wav_bytes)
            except Exception as e:
                traceback.print_exc()
                print(task, str(e))
                batch_results.append(None)
        
        # Return results in order
        for result in batch_results:
            output_queue.put(result)

def create_audiobook_multi(input_queue, output_queue, text_content, voice_library_path, project_name):
    """Create multi-voice audiobook from tagged text with batched processing"""
    if not text_content or not project_name:
        return None, "❌ Missing required fields"
    
    try:
        # Parse and validate the text
        is_valid, message, voice_counts = validate_multi_voice_text(text_content, voice_library_path)
        if not is_valid:
            return None, f"❌ Text validation failed: {message}"
        
        # Get voice segments and chunk them
        segments = parse_multi_voice_text(text_content)
        chunks = chunk_multi_voice_segments(segments, max_words=50)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return None, "❌ No text chunks to process"
        
        # Project directory
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        project_dir = os.path.join("audiobook_projects", safe_project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        print(f"🚀 Starting batch processing of {total_chunks} chunks...")
        
        # Submit all chunks to queue at once
        chunk_configs = []
        for i, (voice_name, chunk_text) in enumerate(chunks, 1):
            # Get voice configuration
            voice_config = get_voice_config(voice_library_path, voice_name)
            if not voice_config:
                return None, f"❌ Could not load voice configuration for '{voice_name}'"
            
            if not voice_config['audio_file']:
                return None, f"❌ No audio file found for voice '{voice_config['display_name']}'"
            
            # Store config for later use
            chunk_configs.append((i, voice_name, voice_config))
            
            print(f"📤 Queuing chunk {i}/{total_chunks} ({voice_name}): {chunk_text[:50]}...")
            
            # Submit to queue
            input_queue.put((
                voice_config['audio_file'], 
                chunk_text, 
                voice_config['infer_timestep'], 
                voice_config['p_w'], 
                voice_config['t_w']
            ))
        
        # Collect results in order
        audio_chunks = []
        print(f"📥 Collecting {total_chunks} results...")
        
        for i, voice_name, voice_config in chunk_configs:
            print(f"⏳ Waiting for chunk {i}/{total_chunks} ({voice_name})...")
            
            result = output_queue.get()  # This will block until result is ready
            
            if result is None:
                return None, f"❌ Error generating chunk {i} with voice '{voice_name}'"
            
            # Save individual chunk with voice name
            chunk_filename = f"{safe_project_name}_{i:03d}_{voice_name}.wav"
            chunk_path = os.path.join(project_dir, chunk_filename)
            
            with open(chunk_path, 'wb') as f:
                f.write(result)
            
            # Load for combining
            with wave.open(chunk_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                audio_chunks.append(audio_data)
            
            print(f"✅ Completed chunk {i}/{total_chunks} ({voice_name})")
        
        # Combine all audio for preview
        combined_audio = np.concatenate(audio_chunks)
        sample_rate = 24000  # Default sample rate
        
        total_words = sum([voice_counts[char] for char in voice_counts.keys()])
        duration_minutes = len(combined_audio) // sample_rate // 60
        
        # Create assignment summary
        assignment_summary = "\n".join([f"🎭 [{char}] → {voice_counts[char]} words" for char in voice_counts.keys()])
        
        success_msg = f"✅ Multi-voice audiobook created successfully!\n📊 {total_words:,} words in {total_chunks} chunks\n🎭 Characters: {len(voice_counts)}\n⏱️ Duration: ~{duration_minutes} minutes\n📁 Saved to: {project_dir}\n🚀 Used batch processing for faster generation\n\nVoice Assignments:\n{assignment_summary}"
        
        return (sample_rate, combined_audio), success_msg
        
    except Exception as e:
        error_msg = f"❌ Error creating multi-voice audiobook: {str(e)}"
        return None, error_msg

def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        inp_audio_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content, latent_file=None)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            output_queue.put(None)

def main(inp_audio, inp_text, infer_timestep, p_w, t_w, processes, input_queue, output_queue):
    print("Push task to the inp queue |", inp_audio, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        # Save the result to a temporary file with .wav extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(res)
            tmp_file.flush()
            return tmp_file.name
    else:
        print("")
        return None

# CSS for better styling
css = """
.voice-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background: #f9f9f9;
}

.tab-nav {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px;
    border-radius: 8px 8px 0 0;
}

.voice-library-header {
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: center;
}

.voice-status {
    background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
    color: white;
    border-radius: 6px;
    padding: 12px;
    margin: 5px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.config-status {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white;
    border-radius: 6px;
    padding: 10px;
    margin: 5px 0;
    font-size: 0.9em;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.audiobook-header {
    background: linear-gradient(90deg, #8b5cf6 0%, #06b6d4 100%);
    color: white;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: center;
}

.file-status {
    background: linear-gradient(135deg, #b45309 0%, #92400e 100%);
    color: white;
    border-radius: 6px;
    padding: 12px;
    margin: 5px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.audiobook-status {
    background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%);
    color: white;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: 500;
}

.instruction-box {
    background: linear-gradient(135deg, #374151 0%, #1f2937 100%) !important;
    color: white !important;
    border-left: 4px solid #3b82f6 !important;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}
"""

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    else:
        devices = None
    
    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    print("Start open workers with batch processing")
    for i in range(num_workers):
        p = mp.Process(target=batch_model_worker, args=(input_queue, output_queue, i % len(devices) if devices is not None else None, 3))
        p.start()
        processes.append(p)

    # Load the saved voice library path
    SAVED_VOICE_LIBRARY_PATH = load_config()

    # Define examples with sample voices and texts
    examples = [
        ["samples/sample-1.mp3", "Hello world, this is a test of speech synthesis using artificial intelligence.", 32, 1.4, 3.0],
        ["samples/sample-2.mp3", "The quick brown fox jumps over the lazy dog in the meadow.", 32, 1.4, 3.0],
        ["samples/sample-3.mp3", "Welcome to MegaTTS3, an advanced text-to-speech system with voice cloning capabilities.", 32, 1.4, 3.0]
    ]

    with gr.Blocks(css=css, title="MegaTTS3 - Voice Library & Audiobook Studio") as demo:
        voice_library_path_state = gr.State(SAVED_VOICE_LIBRARY_PATH)
        
        gr.HTML("""
        <div class="voice-library-header">
            <h1>🎧 MegaTTS3 - Voice Library & Audiobook Studio</h1>
            <p>Advanced text-to-speech with voice cloning and audiobook creation</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Enhanced TTS Tab with Voice Selection
            with gr.TabItem("🎤 Text-to-Speech", id="tts"):
                with gr.Row():
                    with gr.Column():
                        # Voice Selection Section
                        with gr.Group():
                            gr.HTML("<h4>🎭 Voice Selection</h4>")
                            tts_voice_selector = gr.Dropdown(
                                choices=get_voice_choices(SAVED_VOICE_LIBRARY_PATH),
                                label="Choose Voice",
                                value=None,
                                info="Select a saved voice profile or use manual input"
                            )
                            
                            # Voice status display
                            tts_voice_status = gr.HTML(
                                "<div class='voice-status'>📝 Manual input mode - upload your own audio file below</div>"
                            )
                        
                        # Audio input (conditionally visible)
                        inp_audio = gr.Audio(
                            sources=["upload", "microphone"], 
                            type="filepath", 
                            label="Reference Audio File (Manual Input)", 
                            value=None,
                            visible=True
                        )
                        
                        inp_text = gr.Textbox(
                            label="Text to Synthesize", 
                            placeholder="Enter the text you want to convert to speech...",
                            value="Welcome to MegaTTS3, an advanced text-to-speech system with voice cloning capabilities."
                        )
                        
                        with gr.Row():
                            infer_timestep = gr.Number(label="Inference Timesteps", value=32, minimum=1, maximum=100)
                            p_w = gr.Number(label="Intelligibility Weight (p_w)", value=1.4, minimum=0.5, maximum=5.0, step=0.1)
                            t_w = gr.Number(label="Similarity Weight (t_w)", value=3.0, minimum=0.0, maximum=10.0, step=0.1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("Clear")
                            submit_btn = gr.Button("Submit", variant="primary")
                            refresh_voices_btn = gr.Button("🔄 Refresh Voices", size="sm")
                    
                    with gr.Column():
                        output_audio = gr.Audio(label="Synthesized Audio", format="wav")
                        
                        gr.Examples(
                            examples=examples,
                            inputs=[inp_audio, inp_text, infer_timestep, p_w, t_w]
                        )
                        
                        gr.HTML("""
                        <div class="instruction-box">
                            <h4>💡 TTS Tips:</h4>
                            <ul>
                                <li><strong>Voice Selection:</strong> Choose a saved voice for consistent results</li>
                                <li><strong>Reference Audio:</strong> 10-25 seconds of clear speech works best</li>
                                <li><strong>Inference Timesteps:</strong> Higher values = better quality but slower</li>
                                <li><strong>Intelligibility Weight:</strong> Controls clarity vs naturalness</li>
                                <li><strong>Similarity Weight:</strong> Controls how close to reference voice</li>
                            </ul>
                        </div>
                        """)

            # Voice Library Tab
            with gr.TabItem("📚 Voice Library", id="voices"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>🎭 Voice Management</h3>")
                        
                        # Voice Library Settings
                        with gr.Group():
                            gr.HTML("<h4>📁 Library Settings</h4>")
                            voice_library_path = gr.Textbox(
                                value=SAVED_VOICE_LIBRARY_PATH,
                                label="Voice Library Folder",
                                placeholder="Enter path to voice library folder",
                                info="This path will be remembered between sessions"
                            )
                            update_path_btn = gr.Button("💾 Save & Update Library Path", size="sm")
                            
                            # Configuration status
                            config_status = gr.HTML(
                                f"<div class='config-status'>📂 Current library: {SAVED_VOICE_LIBRARY_PATH}</div>"
                            )
                        
                        # Voice Selection
                        with gr.Group():
                            gr.HTML("<h4>🎯 Select Voice</h4>")
                            voice_dropdown = gr.Dropdown(
                                choices=[],
                                label="Saved Voice Profiles",
                                value=None
                            )
                            
                            with gr.Row():
                                load_voice_btn = gr.Button("📥 Load Voice", size="sm")
                                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                                delete_voice_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    
                    with gr.Column(scale=2):
                        # Voice Testing & Saving
                        gr.HTML("<h3>🎙️ Voice Testing & Configuration</h3>")
                        
                        with gr.Group():
                            gr.HTML("<h4>📝 Voice Details</h4>")
                            voice_name = gr.Textbox(label="Voice Name", placeholder="e.g., narrator_male_deep")
                            voice_display_name = gr.Textbox(label="Display Name", placeholder="e.g., Deep Male Narrator")
                            voice_description = gr.Textbox(
                                label="Description", 
                                placeholder="e.g., Deep, authoritative voice for main character",
                                lines=2
                            )
                        
                        with gr.Group():
                            gr.HTML("<h4>🎵 Voice Settings</h4>")
                            voice_audio = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="Reference Audio"
                            )
                            
                            with gr.Row():
                                voice_infer_timestep = gr.Number(
                                    label="Inference Timesteps",
                                    value=32,
                                    minimum=1,
                                    maximum=100
                                )
                                voice_p_w = gr.Number(
                                    label="Intelligibility Weight (p_w)",
                                    value=1.4,
                                    minimum=0.5,
                                    maximum=5.0,
                                    step=0.1
                                )
                                voice_t_w = gr.Number(
                                    label="Similarity Weight (t_w)",
                                    value=3.0,
                                    minimum=0.0,
                                    maximum=10.0,
                                    step=0.1
                                )
                        
                        # Test Voice
                        with gr.Group():
                            gr.HTML("<h4>🧪 Test Voice</h4>")
                            test_text = gr.Textbox(
                                value="Hello, this is a test of the voice settings. How does this sound?",
                                label="Test Text",
                                lines=2
                            )
                            
                            with gr.Row():
                                test_voice_btn = gr.Button("🎵 Test Voice", variant="secondary")
                                save_voice_btn = gr.Button("💾 Save Voice Profile", variant="primary")
                            
                            test_audio_output = gr.Audio(label="Test Audio Output")
                            
                            # Status messages
                            voice_status = gr.HTML("<div class='voice-status'>Ready to test and save voices...</div>")

            # Single Voice Audiobook Creation Tab
            with gr.TabItem("📖 Audiobook Creation - Single Voice", id="audiobook_single"):
                gr.HTML("""
                <div class="audiobook-header">
                    <h2>📖 Audiobook Creation Studio - Single Voice</h2>
                    <p>Transform your text into professional audiobooks with one consistent voice</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Text Input Section
                        with gr.Group():
                            gr.HTML("<h3>📝 Text Content</h3>")
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    audiobook_text = gr.Textbox(
                                        label="Audiobook Text",
                                        placeholder="Paste your text here or upload a file below...",
                                        lines=12,
                                        max_lines=20,
                                        info="Text will be split into chunks at sentence boundaries"
                                    )
                                
                                with gr.Column(scale=1):
                                    text_file = gr.File(
                                        label="📄 Upload Text File",
                                        file_types=[".txt", ".md", ".rtf"],
                                        type="filepath"
                                    )
                                    
                                    load_file_btn = gr.Button(
                                        "📂 Load File", 
                                        size="sm",
                                        variant="secondary"
                                    )
                                    
                                    # File status
                                    file_status = gr.HTML(
                                        "<div class='file-status'>📄 No file loaded</div>"
                                    )
                    
                    with gr.Column(scale=1):
                        # Voice Selection & Project Settings
                        with gr.Group():
                            gr.HTML("<h3>🎭 Voice Configuration</h3>")
                            
                            audiobook_voice_selector = gr.Dropdown(
                                choices=get_audiobook_voice_choices(SAVED_VOICE_LIBRARY_PATH),
                                label="Select Voice",
                                value=None,
                                info="Choose from your saved voice profiles"
                            )
                            
                            refresh_audiobook_voices_btn = gr.Button(
                                "🔄 Refresh Voices", 
                                size="sm"
                            )
                            
                            # Voice info display
                            audiobook_voice_info = gr.HTML(
                                "<div class='voice-status'>🎭 Select a voice to see details</div>"
                            )
                        
                        # Project Settings
                        with gr.Group():
                            gr.HTML("<h3>📁 Project Settings</h3>")
                            
                            project_name = gr.Textbox(
                                label="Project Name",
                                placeholder="e.g., my_first_audiobook",
                                info="Used for naming output files (project_001.wav, project_002.wav, etc.)"
                            )
                
                # Processing Section
                with gr.Group():
                    gr.HTML("<h3>🚀 Audiobook Processing</h3>")
                    
                    with gr.Row():
                        validate_btn = gr.Button(
                            "🔍 Validate Input", 
                            variant="secondary",
                            size="lg"
                        )
                        
                        process_btn = gr.Button(
                            "🎵 Create Audiobook", 
                            variant="primary",
                            size="lg",
                            interactive=False
                        )

                    # Status and progress
                    audiobook_status = gr.HTML(
                        "<div class='audiobook-status'>📋 Ready to create audiobooks! Load text, select voice, and set project name.</div>"
                    )
                    
                    # Preview/Output area
                    audiobook_output = gr.Audio(
                        label="Generated Audiobook (Preview - Full files saved to project folder)",
                        visible=False
                    )
                
                # Instructions
                gr.HTML("""
                <div class="instruction-box">
                    <h4>📋 How to Create Single-Voice Audiobooks:</h4>
                    <ol>
                        <li><strong>Add Text:</strong> Paste text or upload a file</li>
                        <li><strong>Select Voice:</strong> Choose from your saved voice profiles</li>
                        <li><strong>Set Project Name:</strong> This will be used for output file naming</li>
                        <li><strong>Validate:</strong> Check that everything is ready</li>
                        <li><strong>Create:</strong> Generate your audiobook with smart chunking!</li>
                    </ol>
                    <p><strong>🎯 Smart Chunking:</strong> Text is automatically split at sentence boundaries after ~50 words for optimal processing.</p>
                    <p><strong>📁 File Output:</strong> Individual chunks saved as project_001.wav, project_002.wav, etc.</p>
                </div>
                """)

            # Multi-Voice Audiobook Creation Tab
            with gr.TabItem("🎭 Audiobook Creation - Multi-Voice", id="audiobook_multi"):
                gr.HTML("""
                <div class="audiobook-header">
                    <h2>🎭 Multi-Voice Audiobook Creation Studio</h2>
                    <p>Create dynamic audiobooks with multiple character voices using voice tags</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Text Input Section with Voice Tags
                        with gr.Group():
                            gr.HTML("<h3>📝 Multi-Voice Text Content</h3>")
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    multi_audiobook_text = gr.Textbox(
                                        label="Multi-Voice Audiobook Text",
                                        placeholder='Use voice tags like: [narrator] Once upon a time... [character1] "Hello!" said the princess. [narrator] She walked away...',
                                        lines=12,
                                        max_lines=20,
                                        info="Use [voice_name] tags to assign text to different voices"
                                    )
                                
                                with gr.Column(scale=1):
                                    # File upload
                                    multi_text_file = gr.File(
                                        label="📄 Upload Text File",
                                        file_types=[".txt", ".md", ".rtf"],
                                        type="filepath"
                                    )
                                    
                                    load_multi_file_btn = gr.Button(
                                        "📂 Load File", 
                                        size="sm",
                                        variant="secondary"
                                    )
                                    
                                    # File status
                                    multi_file_status = gr.HTML(
                                        "<div class='file-status'>📄 No file loaded</div>"
                                    )
                    
                    with gr.Column(scale=1):
                        # Project Settings
                        with gr.Group():
                            gr.HTML("<h3>📁 Project Settings</h3>")
                            
                            multi_project_name = gr.Textbox(
                                label="Project Name",
                                placeholder="e.g., my_multi_voice_story",
                                info="Used for naming output files (project_001_character.wav, etc.)"
                            )
                            
                            refresh_multi_voices_btn = gr.Button(
                                "🔄 Refresh Available Voices", 
                                size="sm"
                            )
                
                # Processing Section
                with gr.Group():
                    gr.HTML("<h3>🚀 Multi-Voice Processing</h3>")
                    
                    with gr.Row():
                        validate_multi_btn = gr.Button(
                            "🔍 Validate Text & Voices", 
                            variant="secondary",
                            size="lg"
                        )
                        
                        process_multi_btn = gr.Button(
                            "🎵 Create Multi-Voice Audiobook", 
                            variant="primary",
                            size="lg",
                            interactive=False
                        )
                    
                    # Status and progress
                    multi_audiobook_status = gr.HTML(
                        "<div class='audiobook-status'>📋 Step 1: Add text with [voice_name] tags<br/>📋 Step 2: Validate to check all voices exist<br/>📋 Step 3: Create audiobook</div>"
                    )
                    
                    # Preview/Output area
                    multi_audiobook_output = gr.Audio(
                        label="Generated Multi-Voice Audiobook (Preview - Full files saved to project folder)",
                        visible=False
                    )
                
                # Instructions for Multi-Voice
                gr.HTML("""
                <div class="instruction-box">
                    <h4>📋 How to Create Multi-Voice Audiobooks:</h4>
                    <ol>
                        <li><strong>Add Voice Tags:</strong> Use [voice_name] before text for that character</li>
                        <li><strong>Set Project Name:</strong> Used for output file naming</li>
                        <li><strong>Validate:</strong> Check that all referenced voices exist in your library</li>
                        <li><strong>Create:</strong> Generate your multi-voice audiobook!</li>
                    </ol>
                    <h4>🎯 Voice Tag Format:</h4>
                    <p><code>[narrator] The story begins here...</code></p>
                    <p><code>[princess] "Hello there!" she said cheerfully.</code></p>
                    <p><code>[narrator] The mysterious figure walked away.</code></p>
                    <p><strong>📁 File Output:</strong> Files named with character: project_001_narrator.wav, project_002_princess.wav, etc.</p>
                </div>
                """)

        # Event Handlers
        
        # TTS Voice Selection
        tts_voice_selector.change(
            fn=load_voice_for_tts,
            inputs=[voice_library_path_state, tts_voice_selector],
            outputs=[inp_audio, infer_timestep, p_w, t_w, inp_audio, tts_voice_status]
        )

        # Refresh voices in TTS tab
        refresh_voices_btn.click(
            fn=refresh_voice_choices,
            inputs=voice_library_path_state,
            outputs=tts_voice_selector
        )

        # TTS Generation
        submit_btn.click(
            fn=partial(main, processes=processes, input_queue=input_queue, output_queue=output_queue),
            inputs=[inp_audio, inp_text, infer_timestep, p_w, t_w],
            outputs=[output_audio]
        )
        
        clear_btn.click(
            fn=lambda: [None, "", 32, 1.4, 3.0, None],
            outputs=[inp_audio, inp_text, infer_timestep, p_w, t_w, output_audio]
        )

        # Voice Library Functions
        def update_voice_library_path(new_path):
            if not new_path.strip():
                return DEFAULT_VOICE_LIBRARY, "❌ Path cannot be empty, using default", refresh_voice_list(DEFAULT_VOICE_LIBRARY), refresh_voice_choices(DEFAULT_VOICE_LIBRARY), refresh_audiobook_voice_choices(DEFAULT_VOICE_LIBRARY)
            
            ensure_voice_library_exists(new_path)
            save_msg = save_config(new_path)
            
            return (
                new_path,
                save_msg,
                refresh_voice_list(new_path),
                refresh_voice_choices(new_path),
                refresh_audiobook_voice_choices(new_path)
            )

        update_path_btn.click(
            fn=update_voice_library_path,
            inputs=voice_library_path,
            outputs=[voice_library_path_state, config_status, voice_dropdown, tts_voice_selector, audiobook_voice_selector]
        )

        refresh_btn.click(
            fn=lambda path: (refresh_voice_list(path), refresh_voice_choices(path), refresh_audiobook_voice_choices(path)),
            inputs=voice_library_path_state,
            outputs=[voice_dropdown, tts_voice_selector, audiobook_voice_selector]
        )

        load_voice_btn.click(
            fn=load_voice_profile,
            inputs=[voice_library_path_state, voice_dropdown],
            outputs=[voice_audio, voice_infer_timestep, voice_p_w, voice_t_w, voice_status]
        )

        test_voice_btn.click(
            fn=partial(main, processes=processes, input_queue=input_queue, output_queue=output_queue),
            inputs=[voice_audio, test_text, voice_infer_timestep, voice_p_w, voice_t_w],
            outputs=test_audio_output
        )

        save_voice_btn.click(
            fn=save_voice_profile,
            inputs=[
                voice_library_path_state, voice_name, voice_display_name, voice_description,
                voice_audio, voice_infer_timestep, voice_p_w, voice_t_w
            ],
            outputs=voice_status
        ).then(
            fn=lambda path: (refresh_voice_list(path), refresh_voice_choices(path), refresh_audiobook_voice_choices(path)),
            inputs=voice_library_path_state,
            outputs=[voice_dropdown, tts_voice_selector, audiobook_voice_selector]
        )

        delete_voice_btn.click(
            fn=delete_voice_profile,
            inputs=[voice_library_path_state, voice_dropdown],
            outputs=[voice_status, voice_dropdown]
        ).then(
            fn=lambda path: (refresh_voice_choices(path), refresh_audiobook_voice_choices(path)),
            inputs=voice_library_path_state,
            outputs=[tts_voice_selector, audiobook_voice_selector]
        )

        # Single-voice audiobook functions
        load_file_btn.click(
            fn=load_text_file,
            inputs=text_file,
            outputs=[audiobook_text, file_status]
        )

        refresh_audiobook_voices_btn.click(
            fn=refresh_audiobook_voice_choices,
            inputs=voice_library_path_state,
            outputs=audiobook_voice_selector
        )

        validate_btn.click(
            fn=validate_audiobook_input,
            inputs=[audiobook_text, audiobook_voice_selector, project_name],
            outputs=[process_btn, audiobook_status, audiobook_output]
        )

        def create_single_audiobook_wrapper(text_content, voice_library_path, selected_voice, project_name):
            return create_audiobook_single(input_queue, output_queue, text_content, voice_library_path, selected_voice, project_name)

        process_btn.click(
            fn=create_single_audiobook_wrapper,
            inputs=[audiobook_text, voice_library_path_state, audiobook_voice_selector, project_name],
            outputs=[audiobook_output, audiobook_status]
        )

        # Multi-voice audiobook functions
        load_multi_file_btn.click(
            fn=load_text_file,
            inputs=multi_text_file,
            outputs=[multi_audiobook_text, multi_file_status]
        )

        def validate_multi_voice_input(text_content, voice_library_path, project_name):
            """Validate inputs for multi-voice audiobook creation"""
            issues = []
            
            if not project_name or not project_name.strip():
                issues.append("📁 Project name is required")
            
            if text_content and len(text_content.strip()) < 10:
                issues.append("📏 Text is too short (minimum 10 characters)")
            
            # Validate voice parsing
            is_valid, voice_message, voice_counts = validate_multi_voice_text(text_content, voice_library_path)
            
            if not is_valid:
                issues.append(voice_message)
            
            if issues:
                return (
                    gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=False),
                    "❌ Please fix these issues:\n" + "\n".join(f"• {issue}" for issue in issues),
                    gr.Audio(visible=False)
                )
            
            # Show voice breakdown
            voice_breakdown = "\n".join([f"🎭 [{voice}]: {words} words" for voice, words in voice_counts.items()])
            total_words = sum(voice_counts.values())
            
            return (
                gr.Button("🎵 Create Multi-Voice Audiobook", variant="primary", size="lg", interactive=True),
                f"✅ Ready for multi-voice audiobook creation!\n📊 {total_words:,} total words\n📁 Project: {project_name.strip()}\n\n{voice_breakdown}",
                gr.Audio(visible=True)
            )

        validate_multi_btn.click(
            fn=validate_multi_voice_input,
            inputs=[multi_audiobook_text, voice_library_path_state, multi_project_name],
            outputs=[process_multi_btn, multi_audiobook_status, multi_audiobook_output]
        )

        def create_multi_audiobook_wrapper(text_content, voice_library_path, project_name):
            return create_audiobook_multi(input_queue, output_queue, text_content, voice_library_path, project_name)

        process_multi_btn.click(
            fn=create_multi_audiobook_wrapper,
            inputs=[multi_audiobook_text, voice_library_path_state, multi_project_name],
            outputs=[multi_audiobook_output, multi_audiobook_status]
        )

        refresh_multi_voices_btn.click(
            fn=lambda path: f"<div class='voice-status'>🔄 Available voices refreshed from: {path}<br/>📚 Re-validate your text to check voice assignments</div>",
            inputs=voice_library_path_state,
            outputs=multi_audiobook_status
        )

        # Load initial voice lists
        demo.load(
            fn=lambda: refresh_voice_list(SAVED_VOICE_LIBRARY_PATH),
            inputs=[],
            outputs=voice_dropdown
        )
        demo.load(
            fn=lambda: refresh_voice_choices(SAVED_VOICE_LIBRARY_PATH),
            inputs=[],
            outputs=tts_voice_selector
        )
        demo.load(
            fn=lambda: refresh_audiobook_voice_choices(SAVED_VOICE_LIBRARY_PATH),
            inputs=[],
            outputs=audiobook_voice_selector
        )

        # GitHub link at the very bottom
        gr.HTML("<div style='text-align: center; margin-top: 30px; padding: 20px;'><a href='https://github.com/Saganaki22/MegaTTS3-WaveVAE' target='_blank'>Github</a></div>")

    demo.launch(server_name='0.0.0.0', server_port=7929, debug=True)
    for p in processes:
        p.join()
