# [ICLR 2025] DiffSplat

<h4 align="center">

DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation

[Chenguo Lin](https://chenguolin.github.io), [Panwang Pan](https://paulpanwang.github.io), [Bangbang Yang](https://ybbbbt.com), [Zeming Li](https://www.zemingli.com), [Yadong Mu](http://www.muyadong.com)

[![arXiv](https://img.shields.io/badge/arXiv-2501.16764-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.16764)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://chenguolin.github.io/projects/DiffSplat)
[![Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/chenguolin/DiffSplat)


<p>
    <img width="144" src="./assets/_demo/1.gif">
    <img width="144" src="./assets/_demo/2.gif">
    <img width="144" src="./assets/_demo/3.gif">
    <img width="144" src="./assets/_demo/4.gif">
    <img width="144" src="./assets/_demo/5.gif">
</p>
<p>
    <img width="144" src="./assets/_demo/6.gif">
    <img width="144" src="./assets/_demo/7.gif">
    <img width="144" src="./assets/_demo/8.gif">
    <img width="144" src="./assets/_demo/9.gif">
    <img width="144" src="./assets/_demo/10.gif">
</p>
<p>
    <img width="730", src="./assets/_demo/overview.png">
</p>

</h4>

This repository contains the official implementation of the paper: [DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation](https://arxiv.org/abs/2501.16764), which is accepted to ICLR 2025.
DiffSplat is a generative framework to synthesize 3D Gaussian Splats from text prompts & single-view images in 1~2 seconds. It is fine-tuned directly from a pretrained text-to-image diffusion model.

Feel free to contact me (chenguolin@stu.pku.edu.cn) or open an issue if you have any questions or suggestions.


## 🔥 See Also

You may also be interested in our other works:
- [**[arXiv 2507] MoVieS**](https://github.com/chenguolin/MoVieS): a feed-forward model for dynamic 3DGS reconstruction from monocular videos.
- [**[arXiv 2506] PartCrafter**](https://github.com/wgsxm/PartCrafter): a 3D-native DiT that can directly generate 3D objects in multiple parts.


## 📢 News

- **2025-03-06**: Training instructions for DiffSplat and ControlNet are provided.
- **2025-02-11**: Training instructions for GSRecon and GSVAE are provided.
- **2025-02-02**: Inference instructions (text-conditioned & image-conditioned & controlnet) are provided.
- **2025-01-29**: The source code and pretrained models are released. Happy 🐍 Chinese New Year 🎆!
- **2025-01-22**: DiffSplat is accepted to ICLR 2025.


## 📋 TODO

- [x] Provide detailed instructions for inference.
- [x] Provide detailed instructions for GSRecon & GSVAE training.
- [x] Provide detailed instructions for DiffSplat training.


## 🔧 Installation

You may need to modify the specific version of `torch` in `settings/setup.sh` according to your CUDA version.
There are not restrictions on the `torch` version, feel free to use your preferred one.
```bash
git clone https://github.com/chenguolin/DiffSplat.git
cd DiffSplat
bash settings/setup.sh
```

### ▶️ Optional: Containerized quickstart (Docker/RunPod)

If you prefer a batteries-included container with SSH, JupyterLab, and FileBrowser pre-wired, you can use the optional files under `docker/runpod/`.

Build the image (from repo root):

```bash
docker build -f docker/runpod/Dockerfile -t diffsplat-runpod .
```

Run locally with GPU (Docker Desktop + WSL2 on Windows/macOS, or native Linux):

```bash
docker run --gpus all \
    -p 22:22 -p 80:80 -p 9090:9090 -p 9999:9999 \
    -e JUPYTER_PASSWORD=changeme \
    diffsplat-runpod
```

Services inside the container:

- SSH: port 22 (key-based auth; set `PUBLIC_KEY` to inject your key)
- JupyterLab: port 9999 (`--ip=0.0.0.0`, `--no-browser`; use token from logs or `JUPYTER_PASSWORD`)
- FileBrowser: port 9090
- Nginx: port 80 (optional welcome)

On RunPod, map these ports in the UI. HTTP ports (80/9090/9999) will be proxied to `https://<pod>-<port>.proxy.runpod.net/`. SSH is a raw TCP port; connect with:

```bash
ssh -i /path/to/key -p <mapped_port> root@<runpod_host>
```


## 📊 Dataset

- We use [G-Objaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) with about 265K 3D objects and 10.6M rendered images (265K x 40 views, including RGB, normal and depth maps) for `GSRecon` and `GSVAE` training. [Its subset](https://github.com/ashawkey/objaverse_filter) with about 83K 3D objects provided by [LGM](https://me.kiui.moe/lgm) is used for `DiffSplat` training. Their text descriptions are provided by the latest version of [Cap3D](https://huggingface.co/datasets/tiange/Cap3D) (i.e., refined by [DiffuRank](https://arxiv.org/abs/2404.07984)).
- We find the filtering is crucial for the generation quality of `DiffSplat`, and a larger dataset is beneficial for the performance of `GSRecon` and `GSVAE`.
- We store the dataset in an internal HDFS cluster in this project. Thus, the training code can NOT be directly run on your local machine. Please implement your own dataloading logic referring to our provided dataset & dataloader code.


## 🚀 Usage

### 📷 Camera Conventions

The camera and world coordinate systems in this project are both defined in the `OpenGL` convention, i.e., X: right, Y: up, Z: backward. The camera is located at `(0, 0, 1.4)` in the world coordinate system, and the camera looks at the origin `(0, 0, 0)`.
Please refer to [kiuikit camera doc](https://kit.kiui.moe/camera) for visualizations of the camera and world coordinate systems.

### 🤗 Pretrained Models

All pretrained models are available at [HuggingFace🤗](https://huggingface.co/chenguolin/DiffSplat).

| **Model Name**                | **Fine-tined From** | **#Param.** | **Link** | **Note** |
|-------------------------------|---------------------|-------------|----------|----------|
| **GSRecon**                   | From scratch                    | 42M            | [gsrecon_gobj265k_cnp_even4](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsrecon_gobj265k_cnp_even4)         | Feed-forward reconstruct per-pixel 3DGS from 4-view (RGB, normal, coordinate) maps         |
| **GSVAE (SD)**                | [SD1.5 VAE](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)                    | 84M            | [gsvae_gobj265k_sd](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsvae_gobj265k_sd)         |          |
| **GSVAE (SDXL)**              | [SDXL fp16 VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)                    | 84M            | [gsvae_gobj265k_sdxl_fp16](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsvae_gobj265k_sdxl_fp16)         | fp16-fixed SDXL VAE is more robust         |
| **GSVAE (SD3)**               | [SD3 VAE](https://huggingface.co/stabilityai/stable-diffusion-3-medium)                    | 84M            | [gsvae_gobj265k_sd3](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsvae_gobj265k_sd3)         |          |
| **DiffSplat (SD1.5)**            | [SD1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)                    | 0.86B            | Text-cond: [gsdiff_gobj83k_sd15__render](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd15__render)<br> Image-cond: [gsdiff_gobj83k_sd15_image__render](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd15_image__render)         | Best efficiency         |
| **DiffSplat (PixArt-Sigma)** | [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS)                    | 0.61B            | Text-cond: [gsdiff_gobj83k_pas_fp16__render](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_pas_fp16__render)<br> Image-cond: [gsdiff_gobj83k_pas_fp16_image__render](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_pas_fp16_image__render)         | Best Trade-off         |
| **DiffSplat (SD3.5m)**         | [SD3.5 median](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)                    | 2.24B            | Text-cond: [gsdiff_gobj83k_sd35m__render](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd35m__render)<br> Image-cond: [gsdiff_gobj83k_sd35m_image__render](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd35m_image__render)         | Best performance        |
| **DiffSplat ControlNet (SD1.5)**         | From scratch                    | 361M            | Depth: [gsdiff_gobj83k_sd15__render__depth](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd15__render__depth)<br> Normal: [gsdiff_gobj83k_sd15__render__normal](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd15__render__normal)<br> Canny: [gsdiff_gobj83k_sd15__render__canny](https://huggingface.co/chenguolin/DiffSplat/tree/main/gsdiff_gobj83k_sd15__render__canny)         |          |
| **(Optional) ElevEst**                   | [dinov2_vitb14_reg](https://github.com/facebookresearch/dinov2)                    | 86 M            | [elevest_gobj265k_b_C25](https://huggingface.co/chenguolin/DiffSplat/tree/main/elevest_gobj265k_b_C25)         | (Optional) Single-view image elevation estimation        |


### ⚡ Inference

#### 0. Download Pretrained Models

Note that:
- Pretrained weights will download from HuggingFace and stored in `./out`.
- Other pretrained models (such as CLIP, T5, image VAE, etc.) will be downloaded automatically and stored in your HuggingFace cache directory.
- If you face problems in visiting HuggingFace Hub, you can try to set the environment variable `export HF_ENDPOINT=https://hf-mirror.com`.
- `GSRecon` pretrained weights is NOT really used during inference. Only its rendering function is used for visualization.

```bash
python3 download_ckpt.py --model_type [MODEL_TYPE] [--image_cond]

# `MODEL_TYPE`: choose from "sd15", "pas", "sd35m", "depth", "normal", "canny", "elevest"
# `--image_cond`: add this flag for downloading image-conditioned models
```

For example, to download the `text-cond SD1.5-based DiffSplat`:
```bash
python3 download_ckpt.py --model_type sd15
```
To download the `image-cond PixArt-Sigma-based DiffSplat`:
```bash
python3 download_ckpt.py --model_type pas --image_cond
```

#### 1. Text-conditioned 3D Object Generation

Note that:
- Model differences may not be significant for simple text prompts. We recommend using `DiffSplat (SD1.5)` for better efficiency, `DiffSplat (SD3.5m)` for better performance, and `DiffSplat (PixArt-Sigma)` for a better trade-off.
- By default, `export HF_HOME=~/.cache/huggingface`, `export TORCH_HOME=~/.cache/torch`. You can change these paths in `scripts/infer.sh`. SD3-related models require HuggingFace token for downloading, which is expected to be stored in `HF_HOME`.
- Outputs will be stored in `./out/<MODEL_NAME>/inference`.
- Prompt is specified by `--prompt` (e.g., `a_toy_robot`). Please seperate words by `_` and it will be replaced by space in the code automatically.
- If `"gif"` is in `--output_video_type`, the output will be a `.gif` file. Otherwise, it will be a `.mp4` file. If `"fancy"` is in `--output_video_type`, the output video will be in a fancy style that 3DGS scales gradually increase while rotating.
- `--seed` is used for random seed setting. `--gpu_id` is used for specifying the GPU device.
- Use `--half_precision` for `BF16` half-precision inference. It will reduce the memory usage but may slightly affect the quality.

```bash
# DiffSplat (SD1.5)
bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render \
--prompt a_toy_robot --output_video_type gif \
--gpu_id 0 --seed 0 [--half_precision]

# DiffSplat (PixArt-Sigma)
bash scripts/infer.sh src/infer_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16__render \
--prompt a_toy_robot --output_video_type gif \
--gpu_id 0 --seed 0 [--half_precision]

# DiffSplat (SD3.5m)
bash scripts/infer.sh src/infer_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m__render \
--prompt a_toy_robot --output_video_type gif \
--gpu_id 0 --seed 0 [--half_precision]
```

You will get:
| DiffSplat (SD1.5) | DiffSplat (PixArt-Sigma) | DiffSplat (SD3.5m) |
|-------------------------|-------------------------------|-------------------------|
| ![sd15_text](./assets/_demo/a_toy_robot/sd15.gif) | ![pas_text](./assets/_demo/a_toy_robot/pas.gif) | ![sd35m_text](./assets/_demo/a_toy_robot/sd35m.gif) |


**More Advanced Arguments**:
- `--prompt_file`: instead of using `--prompt`, `--prompt_file` will read prompts from a `.txt` file line by line.
- Diffusion configurations:
    - `--scheduler_type`: choose from `ddim`, `dpmsolver++`, `sde-dpmsolver++`, etc.
    - `--num_inference_timesteps`: the number of diffusion steps.
    - `--guidance_scale`: classifier-free guidance (CFG) scale; `1.0` means no CFG.
    - `--eta`: specified for `DDIM` scheduler; the weight of noise for added noise in diffusion steps.
- [Instant3D](https://instant-3d.github.io) tricks:
    - `--init_std`, `--init_noise_strength`, `--init_bg`: initial noise settings, cf. [Instant3D Sec. 3.1](https://arxiv.org/pdf/2311.06214); NOT used by default, as we found it's not that helpful in our case.
- Others:
    - `--elevation`: elevation for viewing and rendering; not necessary for text-conditioned generation; set to `10` by default (from xz-plane (`0`) to +y axis (`90`)).
    - `--negative_prompt`: empty prompt (`""`) by default; used with CFG for better visual quality (e.g., more vibrant colors), but we found it causes lower metric values (such as [ImageReward](https://github.com/THUDM/ImageReward)).
    - `--save_ply`: save the generated 3DGS as a `.ply` file; used with `--opacity_threshold_ply` to filter out low-opacity splats for a much smaller `.ply` file size.
    - `--eval_text_cond`: evaluate text-conditioned generation automatically.
    - ...

Please refer to [infer_gsdiff_sd.py](./src/infer_gsdiff_sd.py), [infer_gsdiff_pas.py](./src/infer_gsdiff_pas.py), and [infer_gsdiff_sd3.py](./src/infer_gsdiff_sd3.py) for more argument details.

#### 2. Image-conditioned 3D Object Generation

Note that:
- Most of the arguments are the same as text-conditioned generation. Our method support **text and image as conditions simultaneously**.
- Elevation is necessary for image-conditioned generation. You can specify the elevation angle by `--elevation` for viewing and rendering (from xz-plane (`0`) to +y axis (`90`)) or estimate it from the input image by `--use_elevest` (download the pretrained `ElevEst` model by `python3 download_ckpt.py --model_type elevest`) first. But we found that the **estimated elevation is not always accurate**, so it's better to set it manually.
- Text prompt is **optional** for image-conditioned generation. If you want to use text prompt, you can specify it by `--prompt` (e.g., `a_frog`), otherwise, empty prompt (`""`) will be used. Note that **DiffSplat (SD3.5m)** is sensitive to text prompts, and it may generate bad results without a proper prompt.
- Remember to set a smaller `--guidance_scale` for image-conditioned generation, as the default value is set for text-conditioned generation. `2.0` is recommended for most cases.
- `--triangle_cfg_scaling` is a trick that set larger CFG values for far-away views from the input image, while smaller CFG values for close-up views, cf. [SV3D Sec. 3](https://arxiv.org/pdf/2403.12008).
- `--rembg_and_center` will remove the background and center the object in the image. It can be used with `--rembg_model_name` (by default `u2net`) and `--border_ratio` (by default `0.2`).
- Image-conditioned generation is more sensitive to arguments, and you may need to tune them for better results.

```bash
# DiffSplat (SD1.5)
bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15_image__render \
--rembg_and_center --triangle_cfg_scaling --output_video_type gif --guidance_scale 2 \
--image_path assets/grm/frog.png --elevation 20 --prompt a_frog

# DiffSplat (PixArt-Sigma)
bash scripts/infer.sh src/infer_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16_image__render \
--rembg_and_center --triangle_cfg_scaling --output_video_type gif --guidance_scale 2 \
--image_path assets/grm/frog.png --elevation 20 --prompt a_frog

# DiffSplat (SD3.5m)
bash scripts/infer.sh src/infer_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m_image__render \
--rembg_and_center --triangle_cfg_scaling --output_video_type gif --guidance_scale 2 \
--image_path assets/grm/frog.png --elevation 20 --prompt a_frog
```

You will get:
| Arguments | DiffSplat (SD1.5) | DiffSplat (PixArt-Sigma) | DiffSplat (SD3.5m) |
|---------|-------------------------|-------------------------------|-------------------------|
| `--elevation 20 --prompt a_frog` | ![sd15_image](./assets/_demo/a_frog/sd15.gif) | ![pas_image](./assets/_demo/a_frog/pas.gif) | ![sd35m_image](./assets/_demo/a_frog/sd35m.gif) |
| `--use_elevest --prompt a_frog` (estimated elevation: -0.78 deg) | ![sd15_image](./assets/_demo/a_frog_elevest/sd15.gif) | ![pas_image](./assets/_demo/a_frog_elevest/pas.gif) | ![sd35m_image](./assets/_demo/a_frog_elevest/sd35m.gif) |
| `--elevation 20` (prompt is `""`) | ![sd15_image](./assets/_demo/a_frog_empty/sd15.gif) | ![pas_image](./assets/_demo/a_frog_empty/pas.gif) | ![sd35m_image](./assets/_demo/a_frog_empty/sd35m.gif) |

**More Advanced Arguments**:
- `--image_dir`: instead of using `--image_path`, `--image_dir` will read images from a directory.

Please refer to [infer_gsdiff_sd.py](./src/infer_gsdiff_sd.py), [infer_gsdiff_pas.py](./src/infer_gsdiff_pas.py), and [infer_gsdiff_sd3.py](./src/infer_gsdiff_sd3.py) for more argument details.

#### 3. ControlNet for 3D Object Generation

Note that:
- After downloading pretrained **DiffSplat (SD1.5)**, you shoule download the controlnet weights by `python3 download_ckpt.py --model_type [depth | normal | canny]`.
- For **depth-controlnet**, values in depth maps are normalized to `[0, 1]` and larger values (white) mean closer to the camera (smaller depth). Please refer to [GObjaverse Dataset](./src/data/gobjaverse_parquet_dataset.py) for more details.
- For **normal-controlnet**, input camera is normalized to locate at `(0, 0, 1.4)` and look at `(0, 0, 0)`, thus the input normal maps are transformed accordingly. Please refer to [GObjaverse Dataset](./src/data/gobjaverse_parquet_dataset.py) for more details.
- For **canny-controlnet**, canny edges are extracted from the input RGB images automatically by `cv2.Canny`. Please refer to [GObjaverse Dataset](./src/data/gobjaverse_parquet_dataset.py) for more details.

```bash
# ControlNet (depth)
bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render \
--load_pretrained_controlnet gsdiff_gobj83k_sd15__render__depth \
--output_video_type gif --image_path assets/diffsplat/controlnet/toy_depth.png \
--prompt teddy_bear --elevation 10

# ControlNet (normal)
bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render \
--load_pretrained_controlnet gsdiff_gobj83k_sd15__render__normal \
--output_video_type gif --image_path assets/diffsplat/controlnet/robot_normal.png \
--prompt iron_robot --elevation 10

# ControlNet (canny)
bash scripts/infer.sh src/infer_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render \
--load_pretrained_controlnet gsdiff_gobj83k_sd15__render__canny \
--output_video_type gif --image_path assets/diffsplat/controlnet/cookie_canny.png \
--prompt book --elevation 10
```

You will get:
| Original Image | Input Control | `--prompt teddy_bear` | `--prompt panda` |
|----------------|---------------|-----------------------|--------------------|
| ![depth_image](./assets/diffsplat/controlnet/toy_image.png) | ![depth](./assets/diffsplat/controlnet/toy_depth.png) | ![controlnet_1](assets/_demo/controlnet/teddy_bear.gif) | ![controlnet_2](assets/_demo/controlnet/panda.gif) |

| Original Image | Input Control | `--prompt iron_robot` | `--prompt plush_dog_toy` |
|----------------|---------------|-----------------------|--------------------|
| ![normal_image](./assets/diffsplat/controlnet/robot_image.png) | ![normal](./assets/diffsplat/controlnet/robot_normal.png) | ![controlnet_1](assets/_demo/controlnet/iron_robot.gif) | ![controlnet_2](assets/_demo/controlnet/plush_dog_toy.gif) |

| Original Image | Input Control | `--prompt book` | `--prompt cookie` |
|----------------|---------------|-----------------|---------------------|
| ![canny_image](./assets/diffsplat/controlnet/cookie_image.png) | ![canny](./assets/diffsplat/controlnet/cookie_canny.png) | ![controlnet_1](assets/_demo/controlnet/book.gif) | ![controlnet_2](assets/_demo/controlnet/cookie.gif) |

**More Advanced Arguments**:
- `--guess_mode`: ControlNet encoder tries to recognize the content of the input image even if you remove all prompts, cf. [the original ControlNet repo](https://github.com/lllyasviel/ControlNet#guess-mode--non-prompt-mode) and [HF ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet#guess-mode).
- `--controlnet_scale`: determines how much weight to assign to the conditioning inputs; outputs of the ControlNet are multiplied by `controlnet_scale` before they are added to the residual in the original UNet.

Please refer to [infer_gsdiff_sd.py](./src/infer_gsdiff_sd.py) for more argument details.


### 🦾 Training

#### 0. Project Overview

##### 0.1 `extensions/diffusers_diffsplat`
We manually modified the latest `diffusers` library (`diffusers==0.32`) and tried to comment in detail on codes to clarify modifications. The folder structure is the same as the original repo.

Generally:
- Modifications in `diffusers_diffsplat/models` are most for (1) "multi-view attention" that gets inputs in `(B*V, N, D)` then operates the attention operation in `(B, V*N, D)`, (2) a new function `from_pretrained_new()` for UNet and Transformer that initializes models with different input channels, e.g., 4 (SD latent) in the original Stable Diffusion, while 10 or 11 (RGB + plucker + (optional) binary mask) for our DiffSplat.
- `diffusers_diffsplat/pipelines` are implemented for each base model in `diffusers_diffsplat/models` accordingly with some fancy functions (such as Instant3D-style noise initialization), which are however not really used.
- `diffusers_diffsplat/schedulers` are only for `DPM-Solver++ flow matching scheduler`, which is copied from [the diffusers PR](https://github.com/huggingface/diffusers/pull/9982) and not really used.
- `diffusers_diffsplat/training_utils.py` is only for `EMAModel` that can really set `self.use_ema_warmup` as described in [the diffusers PR](https://github.com/huggingface/diffusers/pull/9812).

##### 0.2 `src/data/gobjaverse_parquet_dataset.py`

We preprocess the original GObjaverse dataset and store it in a parquet format for efficient dataloading from an internal HDFS. The parquet format is NOT necessary, and you can implement your own dataloading logic.

Here is our preprocessing script for your reference:
```python
# Code snippet for GObjaverse dataset preprocessing; NOT runnable
# ...

outputs = {
    "__key__": objaverse_id,
    "uid": objaverse_id.encode("utf-8"),
}

dir_id, object_id = item.split("/")
object_dir = os.path.join(SAVE_DIR, dir_id, object_id, "campos_512_v4")

for i in range(40):  # hard-coded `40` views
    view_dir = os.path.join(object_dir, f"{i:05}")
    image_path = os.path.join(view_dir, f"{i:05}.png")
    albedo_path = os.path.join(view_dir, f"{i:05}_albedo.png")
    mr_path = os.path.join(view_dir, f"{i:05}_mr.png")
    nd_path = os.path.join(view_dir, f"{i:05}_nd.exr")
    transform_path = os.path.join(view_dir, f"{i:05}.json")

    # Use `tf.io.encode_png` to encode images to compact bytes
    try:
        outputs[f"{i:05}.png"] = tf.io.encode_png(tf.convert_to_tensor(imageio.imread(image_path), tf.uint8)).numpy()
        outputs[f"{i:05}_albedo.png"] = tf.io.encode_png(tf.convert_to_tensor(imageio.imread(albedo_path)[:, :, :3], tf.uint8)).numpy()
        outputs[f"{i:05}_mr.png"] = tf.io.encode_png(tf.convert_to_tensor(imageio.imread(mr_path)[:, :, :3], tf.uint8)).numpy()
        nd = cv2.imread(nd_path, cv2.IMREAD_UNCHANGED)
        nd[:, :, :3] = nd[:, :, :3][..., ::-1]  # BGR -> RGB
        nd[:, :, :3] = (nd[:, :, :3] * 0.5 + 0.5) * 65535  # [-1., 1.] -> [0, 65535]
        nd[:, :, 3] = nd[:, :, 3] / 5. * 65535  # scale the depth by 1/5, then it must be in [0, 1]; [0., +?) -> [0, 65535]
        outputs[f"{i:05}_nd.png"] = tf.io.encode_png(tf.convert_to_tensor(nd, tf.uint16)).numpy()
        with open(transform_path, "r") as f:
            outputs[f"{i:05}.json"] = f.read().encode("utf-8")

        if outputs[f"{i:05}.png"] is None or outputs[f"{i:05}_albedo.png"] is None or \
            outputs[f"{i:05}_mr.png"] is None or outputs[f"{i:05}_nd.png"] is None or \
            outputs[f"{i:05}.json"] is None:
            continue  # ignore broken files
    except:
        continue  # ignore broken files

    # Then `outputs: Dict[str, bytes]` is stored in a parquet file
    # ...
```

#### 1. GSRecon

Set environment variables in `scripts/train.sh` first, then:
```bash
bash scripts/train.sh src/train_gsrecon.py configs/gsrecon.yaml gsrecon_gobj265k_cnp_even4
```

Please refer to [train_gsrecon.py](./src/train_gsrecon.py) and options are specified in [configs/gsrecon.yaml](./configs/gsrecon.yaml) and [options.py](./src/options.py) (`opt_dict["gsrecon"]`).

Please refer to [issues#12](https://github.com/chenguolin/DiffSplat/issues/12) to infer `GSRecon` with multi-view (4 views) RGB, normal, and coordinate maps.

#### 2. GSVAE

##### 2.1 Regular GSVAE

Set environment variables in `scripts/train.sh` first, then:
```bash
# SD1.5 / SD2.1 / PixArt-alpha
bash scripts/train.sh src/train_gsvae.py configs/gsvae.yaml gsvae_gobj265k_sd opt_type=gsvae --gradient_accumulation_steps 4

# SDXL (fp16-fixed) / PixArt-Sigma
bash scripts/train.sh src/train_gsvae.py configs/gsvae.yaml gsvae_gobj265k_sdxl_fp16 opt_type=gsvae_sdxl_fp16 --gradient_accumulation_steps 4

# SD3 / SD3.5
bash scripts/train.sh src/train_gsvae.py configs/gsvae.yaml gsvae_gobj265k_sd3 opt_type=gsvae_sd35m --gradient_accumulation_steps 4
```

##### 2.2 Tiny GSVAE Decoder

For efficient performing rendering loss in the DiffSplat training stage, we train **tiny decoders** (use pretrained tiny AEs at [here](https://huggingface.co/madebyollin)) with much smaller sizes than the original decoders.
Note that:
- Tiny GSVAE decoders are only used in DiffSplat rendering loss, not for the final inference.
- `opt.freeze_encoder=true`: encoder part of the pretrained GSVAE is fixed.
- `opt.use_tiny_decoder=true`: use tiny decoder in this stage.
- `--load_pretrained_model`: load pretrained GSVAE models in the previous stage.

Set environment variables in `scripts/train.sh` first, then:
```bash
# Tiny SD1.5 / SD2.1 / PixArt-alpha decoder
bash scripts/train.sh src/train_gsvae.py configs/gsvae.yaml gsvae_gobj265k_sd opt_type=gsvae train.batch_size_per_gpu=8 opt.freeze_encoder=true opt.use_tiny_decoder=true --load_pretrained_model gsvae_gobj265k_sd

# Tiny SDXL (fp16-fixed) / PixArt-Sigma decoder
bash scripts/train.sh src/train_gsvae.py configs/gsvae.yaml gsvae_gobj265k_sdxl_fp16 opt_type=gsvae_sdxl_fp16 train.batch_size_per_gpu=8 opt.freeze_encoder=true opt.use_tiny_decoder=true --load_pretrained_model gsvae_gobj265k_sdxl_fp16

# Tiny SD3 / SD3.5 decoder
bash scripts/train.sh src/train_gsvae.py configs/gsvae.yaml gsvae_gobj265k_sd3 opt_type=gsvae_sd35m train.batch_size_per_gpu=8 opt.freeze_encoder=true opt.use_tiny_decoder=true --load_pretrained_model gsvae_gobj265k_sd3
```

Please refer to [train_gsvae.py](./src/train_gsvae.py) and options are specified in [configs/gsvae.yaml](./configs/gsvae.yaml) and [options.py](./src/options.py) (`opt_dict["gsvae"]`, `opt_dict["gsvae_sdxl_fp16"]` and `opt_dict["gsvae_sd35m"]`).

#### 3. DiffSplat

##### 3.0 Text Embedding Precomputation

Text embeddings for captions are precomputed by [extensions/encode_prompt_embeds.py](./extensions/encode_prompt_embeds.py):
```bash
python3 extensions/encode_prompt_embeds.py [MODEL_NAME] [--batch_size 128] [--dataset_name gobj83k]

# `MODEL_NAME`: choose from "sd15", "sd21", "sdxl", "paa", "pas", "sd3m", "sd35m", "sd35l"
```
Captions will download automatically in `extensions/assets` and text embeddings are stored in `/tmp/{DATASET_NAME}_{MODEL_NAME}_prompt_embeds` by default.

##### 3.1 DiffSplat (w/o rendering loss)
Note that:
- `opt.view_concat_condition=true opt.input_concat_binary_mask=true`: specified for image-conditioned generation.
- `opt.prediction_type=v_prediction`: specified for image-conditioned generation. We use `v_prediction` for better image-conditioned performance.
- `----val_guidance_scales 1 2 3` (default: `1 3 7.5`): smaller CFG scales for image conditioning.

Set environment variables in `scripts/train.sh` first, then:
```bash
# SD1.5 (text-cond)
bash scripts/train.sh src/train_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15 --gradient_accumulation_steps 2 --use_ema
# SD1.5 (image-cond)
bash scripts/train.sh src/train_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15 --gradient_accumulation_steps 2 --use_ema ----val_guidance_scales 1 2 3 opt.view_concat_condition=true opt.input_concat_binary_mask=true opt.prediction_type=v_prediction

# PixArt-Sigma (text-cond)
bash scripts/train.sh src/train_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16 --gradient_accumulation_steps 2 --use_ema
# PixArt-Sigma (image-cond)
bash scripts/train.sh src/train_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16 --gradient_accumulation_steps 2 --use_ema ----val_guidance_scales 1 2 3 opt.view_concat_condition=true opt.input_concat_binary_mask=true opt.prediction_type=v_prediction

# SD3.5m (text-cond)
bash scripts/train.sh src/train_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m --gradient_accumulation_steps 8 --use_ema
# SD3.5m (image-cond)
bash scripts/train.sh src/train_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m --gradient_accumulation_steps 8 --use_ema ----val_guidance_scales 1 2 3 opt.view_concat_condition=true opt.input_concat_binary_mask=true opt.prediction_type=v_prediction
```

##### 3.2 DiffSplat (w/ rendering loss)
Note that:
- `opt.rendering_loss_prob=1` (default `0`) means use rendering loss in the training stage all the time.
- `opt.snr_gamma_rendering=1`: we use SNR (signal-noise ratio) weighted rendering loss (weight `gamma=1`) for PixArt-Sigma models for more robust training. Feel free to tune this weight for other models.
- `opt.use_tiny_decoder=true`: use tiny decoder for efficient decoding/rendering in this stage.
- `--load_pretrained_model` is used for loading pretrained DiffSplat models in the previous stage.

Set environment variables in `scripts/train.sh` first, then:
```bash
# SD1.5 (text-cond)
bash scripts/train.sh src/train_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render --gradient_accumulation_steps 2 --use_ema opt.rendering_loss_prob=1 opt.use_tiny_decoder=true --load_pretrained_model gsdiff_gobj83k_sd15
# SD1.5 (image-cond)
bash scripts/train.sh src/train_gsdiff_sd.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15_image__render --gradient_accumulation_steps 2 --use_ema ----val_guidance_scales 1 2 3 opt.view_concat_condition=true opt.input_concat_binary_mask=true opt.prediction_type=v_prediction opt.rendering_loss_prob=1 opt.use_tiny_decoder=true --load_pretrained_model gsdiff_gobj83k_sd15

# PixArt-Sigma (text-cond)
bash scripts/train.sh src/train_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16__render --gradient_accumulation_steps 2 --use_ema opt.rendering_loss_prob=1 opt.snr_gamma_rendering=1 opt.use_tiny_decoder=true --load_pretrained_model gsdiff_gobj83k_pas_fp16
# PixArt-Sigma (image-cond)
bash scripts/train.sh src/train_gsdiff_pas.py configs/gsdiff_pas.yaml gsdiff_gobj83k_pas_fp16_image__render --gradient_accumulation_steps 2 --use_ema ----val_guidance_scales 1 2 3 opt.view_concat_condition=true opt.input_concat_binary_mask=true opt.prediction_type=v_prediction opt.rendering_loss_prob=1 opt.snr_gamma_rendering=1 opt.use_tiny_decoder=true --load_pretrained_model gsdiff_gobj83k_pas_fp16

# SD3.5m (text-cond)
bash scripts/train.sh src/train_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m__render --gradient_accumulation_steps 8 --use_ema opt.rendering_loss_prob=1 opt.use_tiny_decoder=true --load_pretrained_model gsdiff_gobj83k_sd35m
# SD3.5m (image-cond)
bash scripts/train.sh src/train_gsdiff_sd3.py configs/gsdiff_sd35m_80g.yaml gsdiff_gobj83k_sd35m_image__render --gradient_accumulation_steps 8 --use_ema ----val_guidance_scales 1 2 3 opt.view_concat_condition=true opt.input_concat_binary_mask=true opt.prediction_type=v_prediction opt.rendering_loss_prob=1 opt.use_tiny_decoder=true --load_pretrained_model gsdiff_gobj83k_sd35m
```

Please refer to [train_gsdiff_{sd, sdxl, paa, pas, sd3}.py](./src/train_gsdiff_sd.py) and options are specified in [configs/gsdiff_{sd, sdxl_80g, paa,pas, sd3m_80g, sd35m_80g}.yaml](./configs/gsdiff_sd15.yaml) and [options.py](./src/options.py) (`opt_dict["gsdiff_sd15"]`, `opt_dict["gsdiff_sdxl"]`, `opt_dict["gsdiff_paa"]`, `opt_dict["gsdiff_pas"]`, `opt_dict["gsdiff_sd3m"]` and `opt_dict["gsdiff_sd35m"]`).

#### 4. ControlNet
Note that:
- `opt.controlnet_type`: choose from `[normal, canny, depth]`.
- `--load_pretrained_model` is used for loading pretrained DiffSplat models in the previous stage.

Set environment variables in `scripts/train.sh` first, then:
```bash
# Normal ControlNet
bash scripts/train.sh src/train_gsdiff_sd_controlnet.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render__normal --gradient_accumulation_steps 2 opt.controlnet_type=normal --load_pretrained_model gsdiff_gobj83k_sd15__render

# Canny ControlNet
bash scripts/train.sh src/train_gsdiff_sd_controlnet.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render__canny --gradient_accumulation_steps 2 opt.controlnet_type=canny --load_pretrained_model gsdiff_gobj83k_sd15__render

# Depth ControlNet
bash scripts/train.sh src/train_gsdiff_sd_controlnet.py configs/gsdiff_sd15.yaml gsdiff_gobj83k_sd15__render__depth --gradient_accumulation_steps 2 opt.controlnet_type=depth --load_pretrained_model gsdiff_gobj83k_sd15__render
```

Please refer to [train_gsdiff_{sd, sdxl}_controlnet.py](./src/train_gsdiff_sd_controlnet.py) and options are in [configs/gsdiff_{sd15, sdxl}_controlnet.yaml](./configs/gsdiff_sd15_controlnet.yaml) and [options.py](./src/options.py) (`opt_dict["gsdiff_sd15"]` and `opt_dict["gsdiff_sdxl"]`).


## 😊 Acknowledgement
We would like to thank the authors of [LGM](https://me.kiui.moe/lgm), [GRM](https://justimyhxu.github.io/projects/grm), and [Wonder3D](https://www.xxlong.site/Wonder3D) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.


## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{lin2025diffsplat,
  title={DiffSplat: Repurposing Image Diffusion Models for Scalable 3D Gaussian Splat Generation},
  author={Lin, Chenguo and Pan, Panwang and Yang, Bangbang and Li, Zeming and Mu, Yadong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
