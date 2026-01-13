import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(10)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_example():
    """ CosyVoice3 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    for i in tqdm(range(10)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_spk_example():
    """
    CosyVoice3 vLLM Usage with pre-generated speaker info .pt file.
    
    This example demonstrates how to use a pre-generated spk2info.pt file for inference,
    which avoids repeated audio processing and speeds up inference.
    
    To generate the .pt file, use spk_example.py:
        python spk_example.py --wav_path ./asset/zero_shot_prompt.wav \
                              --prompt_text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
                              --spk_name "my_speaker" \
                              --output_path ./my_speaker.pt \
                              --model_dir pretrained_models/Fun-CosyVoice3-0.5B
    """
    import torch
    import torchaudio
    
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    
    # Method 1: Load .pt file and add speaker to frontend.spk2info
    spk2info = torch.load('./spkinfo.pt', map_location='cpu')
    spk_name = list(spk2info.keys())[0]  # Get the speaker name from the .pt file
    cosyvoice.frontend.spk2info[spk_name] = spk2info[spk_name]
    
    # Now use inference_zero_shot with zero_shot_spk_id (no need for prompt_text and prompt_wav)
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        'You are a helpful assistant.<|endofprompt|>',  # Not used when zero_shot_spk_id is set; keeps CosyVoice3 prompt format consistent
        '',  # Empty prompt_wav since we use cached spk_info
        zero_shot_spk_id=spk_name,
        stream=False
    )):
        torchaudio.save('vllm_spk_example_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    
    print(f"Generated speech using pre-loaded speaker '{spk_name}'")
    
    # Method 2: Directly use spk_info in frontend for cross_lingual inference
    for i, j in enumerate(cosyvoice.inference_cross_lingual(
        'You are a helpful assistant.<|endofprompt|>Hello, this is a test of cross-lingual synthesis.',
        '',  # Empty prompt_wav
        zero_shot_spk_id=spk_name,
        stream=False
    )):
        torchaudio.save('vllm_spk_example_cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    
    print(f"Generated cross-lingual speech using pre-loaded speaker '{spk_name}'")


def main():
    # cosyvoice2_example()
    cosyvoice3_example()
    cosyvoice3_spk_example()


if __name__ == '__main__':
    main()
