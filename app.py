import os

import sys
from torchvision.transforms import functional
sys.modules["torchvision.transforms.functional_tensor"] = functional

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

import torch
import cv2
import gradio as gr


#Download Required Models
if not os.path.exists('realesr-general-x4v3.pth'):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
if not os.path.exists('GFPGANv1.2.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P .")
if not os.path.exists('GFPGANv1.3.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .")
if not os.path.exists('GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('RestoreFormer.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P .")


model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)


# Save Image to the Directory
# os.makedirs('output', exist_ok=True)

def upscaler(img, version, scale):

    try:
        
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None


        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        
        face_enhancer = GFPGANer(
            model_path=f'{version}.pth', 
            upscale=2, 
            arch='RestoreFormer' if version=='RestoreFormer' else 'clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )


        try:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error', error)


        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('wrong scale input.', error)

        # Save Image to the Directory
        # ext = os.path.splitext(os.path.basename(str(img)))[1]
        # if img_mode == 'RGBA':
        #     ext = 'png'
        # else:
        #     ext = 'jpg'
        #
        # save_path = f'output/out.{ext}'
        # cv2.imwrite(save_path, output)
        # return output, save_path

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output
    except Exception as error:
        print('global exception', error)
        return None, None

if __name__ == "__main__":

    title = "Image Upscaler & Restoring [GFPGAN Algorithm]"

    demo = gr.Interface(
            upscaler, [
                gr.Image(type="filepath", label="Input"),
                gr.Radio(['GFPGANv1.2', 'GFPGANv1.3', 'GFPGANv1.4', 'RestoreFormer'], type="value", label='version'),
                gr.Number(label="Rescaling factor"),
            ], [
                gr.Image(type="numpy", label="Output"),
            ],
            title=title,
            allow_flagging="never"
        )

    demo.queue()
    demo.launch()