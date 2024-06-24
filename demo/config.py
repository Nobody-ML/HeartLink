from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig
import os

#######################################################################
#                          PART 1  lmdeploy                           #
#######################################################################
SYSTEM = os.getenv("SYSTEM")

IS_TURBOMIND = True
IS_PYTORCH = False

backend_config = TurbomindEngineConfig(cache_max_entry_count=0.3)
chat_template_config = ChatTemplateConfig(model_name='internlm2',meta_instruction=SYSTEM)

#######################################################################
#                          PART 2  TTS                                #
#######################################################################
prompt_text = "胡桃的胡是胡吃海喝的胡，胡桃的桃却不是淘气的淘！嘿嘿…不、不好笑吗？"
prompt_language = "中文"
text_language = "中文"
ref_wav_path = "/home/xlab-app-center/demo/TTS/GPT_SoVITS/cankao2.wav"
