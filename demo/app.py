import sys
import streamlit as st
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import torch
import lmdeploy
import json
import os
import time
import soundfile as sf
import copy
import pandas as pd
import altair as alt
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataclasses import asdict, dataclass
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig

from TTS.GPT_SoVITS.utils import HParams

from config import backend_config,chat_template_config,IS_TURBOMIND
from config import prompt_text, prompt_language, text_language, ref_wav_path
from TTS.GPT_SoVITS.tts import get_tts_wav, load_tts_model
from TTS.GPT_SoVITS.feature_extractor import cnhubert
from modelscope.hub.api import HubApi

os.system("pwd")
print("-------------")
os.system("ls")
print("-------------")

os.system("mv /home/xlab-app-center/nltk_data /home/xlab-app-center")
api = HubApi()
api.login('3495b435-5eb0-41c8-89eb-254c8c971b4e')

from modelscope import snapshot_download
model_dir1 = snapshot_download('NobodyYing/HeartLink_7B_qlora_analyse', cache_dir='/home/xlab-app-center')

model_dir2 = snapshot_download('NobodyYing/GPT_SoVITS_pretrained_models', cache_dir='/home/xlab-app-center')

model_dir3 = snapshot_download('NobodyYing/GPT_weights_hutao', cache_dir='/home/xlab-app-center')

model_dir4 = snapshot_download('NobodyYing/SoVITS_weights_hutao', cache_dir='/home/xlab-app-center')

gradient_text_html = """
<style>
.container {
    position: relative;
    /* 可能需要调整的高度，以避免内容重叠 */
    padding-top: 50px; 
}

.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
    /* 使用相对定位并上移 */
    position: relative;
    top: -105px;
}
</style>
<div class="container">
    <div class="gradient-text">HeartLink</div>
</div>
"""
st.markdown(gradient_text_html, unsafe_allow_html=True)

def on_btn_click():
    del st.session_state.messages

def turbomind_generation_config():
    with st.sidebar:
        st.title("HeartLink——共情大模型")
        st.subheader("目前支持功能")
        st.markdown("- 💖 共情对话")
        st.markdown("- 💬 语音生成(胡桃)")
        st.markdown("- 📊 情绪分析")
        with st.container(height=200, border=True):
            st.subheader("模型配置")
            max_length = st.slider('Max Length',
                                min_value=8,
                                max_value=4096,
                                value=4096)
            top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
            temperature = st.slider('Temperature', 0.0, 1.0, 0.8, step=0.01)
        
        st.button('清空历史对话', on_click=on_btn_click)

    tb_generation_config = GenerationConfig(top_p=top_p,
                                         temperature=temperature,
                                         max_new_tokens=max_length,)

    return tb_generation_config

@st.cache_resource
def load_llm_model():
    if IS_TURBOMIND:
        pipe = pipeline('/home/xlab-app-center/NobodyYing/HeartLink_7B_qlora_analyse',
                    backend_config=backend_config,
                    chat_template_config=chat_template_config,
                    )
    return pipe


def llm_prompt():
    prompts = []
    mes = copy.deepcopy(st.session_state.messages)
    for detail in mes:
        if detail["role"] == "robot" or detail["role"] == "assistant":
            detail["role"] = "assistant"
            del detail['wav'] 
            del detail['emotions']
            del detail['avatar']
        prompts.append(detail)
    return prompts

def main():
    print('load llm')
    pipe = load_llm_model()
    print('load llm done')

    print('load tts model')
    tokenizer, bert_model, ssl_model, vq_model, hps, t2s_model, max_sec = load_tts_model()
    print('load tts model done')

    tb_generation_config = turbomind_generation_config()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # with st.container():
    #     col1, col2 = st.columns([3, 2])

    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            if message['role'] == 'robot' or message['role'] == 'assistant':
                try:
                    content = json.loads(message['content'])["共情回复"]
                except (json.JSONDecodeError, KeyError):
                    content = message['content']
                st.markdown(content)
            else:
                st.markdown(message['content'])
            
            if message.get("wav") is not None:
                with open(message["wav"], "rb") as wav:
                    audio = wav.read()
                st.audio(audio, format="audio/wav")
    
    if prompt := st.chat_input('请告诉我你的经历与感受～'):
        with st.chat_message('user',):
                st.markdown(prompt)

        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        prompts = llm_prompt()
        with st.chat_message('robot',avatar='/home/xlab-app-center/demo/asserts/logo.jpg'):
            message_placeholder = st.empty()
            loading_placeholder = st.empty()
            # border,width,height调圈大小，<div style="display: flex; align-items: center; margin-top: -15px;">加justify-content: center;居中
            loading_placeholder.markdown("""
                <div style="display: flex; align-items: center; margin-top: -15px;">
                    <div class="spinner"></div>
                    <span style="margin-left: 10px;">正在生成文本，请稍等</span>
                </div>
                <style>
                    .spinner {
                        border: 2px solid rgba(0, 0, 0, 0.1);
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        border-left-color: #09f;
                        animation: spin 1s ease infinite;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            """, unsafe_allow_html=True)
            items = ''
            print(st.session_state.messages)
            while True:
                for item in pipe.stream_infer(prompts=prompts, gen_config=tb_generation_config):
                    items += item.text
                    print(item.text,end='')
                try:
                    response = json.loads(items)["共情回复"]
                    emotion = json.loads(items)["情绪"].replace("，" ,",")
                    break
                except:
                    continue

            loading_placeholder.empty()
            message_placeholder.markdown(response)
            
            with st.spinner("正在生成语音，请稍等～"):
                sr, audio = get_tts_wav(ref_wav_path=ref_wav_path, prompt_text=prompt_text, prompt_language=prompt_language, text=response, text_language=text_language, 
                                        tokenizer=tokenizer, bert_model=bert_model, ssl_model=ssl_model, vq_model=vq_model, hps=hps, t2s_model=t2s_model, max_sec=max_sec,
                                        )
                
                output_wav_path = "/home/xlab-app-center/demo/TTS/tts_temp/"
                now_time = time.time()
                
                sf.write(output_wav_path+str(now_time)+'.wav', audio, sr)
                with open(output_wav_path+str(now_time)+'.wav', "rb") as wav:
                    audio = wav.read()
                try:
                    st.audio(data=audio, format="audio/wav", autoplay=True)
                except:
                    st.audio(data=audio, format="audio/wav")

            emotions = emotion.split(',')

            try:
                print(st.session_state.messages)
                tmp = copy.deepcopy(st.session_state.messages[-2]["emotions"])
                for e in emotions:
                    try:
                        tmp[e] += 1
                    except:
                        tmp[e] = 1

            except:
                print(st.session_state.messages)
                print('false')
                tmp = {e: 1 for e in emotions}
            print(tmp)

            st.session_state.messages.append({
                'role': 'assistant',
                'content': items,
                'wav': output_wav_path+str(now_time)+'.wav',
                'emotions': tmp,
                'avatar': '/home/xlab-app-center/demo/asserts/logo.jpg',
            })

            
            with st.sidebar:
                st.subheader("情绪分析图表")
                df = pd.DataFrame(list(tmp.items()), columns=['Emotion', 'Count'])
                chart = alt.Chart(df).mark_bar(size=50).encode(
                    x=alt.X('Count:Q', title='Count'),
                    y=alt.Y('Emotion:N', title='Emotion', axis=alt.Axis(labelAngle=0)),  # 设置x轴标签横向显示
                    
                    color=alt.Color('Emotion:N', legend=None)
                ).properties(
                    width=400,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
