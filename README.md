# Interactive painter to experiment with oriented and non-oriented mean curvature models found by machine learning

![nnpf](https://user-images.githubusercontent.com/10435058/207285788-97c68200-e19d-4657-9d94-abda977152dc.png)

For the original article about the associated models and results, see [Learning phase field mean curvature flows with neural networks, Bretin & Denis & Masnou & Terii, 2022](https://www.sciencedirect.com/science/article/pii/S0021999122006416) and on [arXiv](https://arxiv.org/abs/2112.07343).
It relies on [PyTorch](https://pytorch.org/) and [Lightning](https://www.pytorchlightning.ai/) throught the dedicated package [nnpf](https://pypi.org/project/nnpf/) (see also the [GitHub repository](https://github.com/PhaseFieldICJ/nnpf)).

## Requirements
You need at least a Python 3.8 and the [nnpf](https://pypi.org/project/nnpf/) package available on Pypi:
```bash
pip install nnpf
```

## Models
Three models are available:
- for an oriented isotropic mean-curvature flow,
- for an non-oriented isotropic mean-curvature flow,
- for an non-oriented anisotropic mean-curvature flow (distance from l⁴ norm).

The models code are in models.py`.

The checkpoints are available in `logs` of the main branch of this repository.
If you want to train it by yourself, you can checkout the `no_data` branch instead or remove the `logs` folder and launch the training using:
```bash
make
```

If you want to use a GPU:
```bash
make options="--gpus=1"
```

And if your GPU has enough memory to store the whole dataset (~2.5GiB):
```bash
make options="--gpus=1 --force_gpu"
```

## Launching

You need to specify the checkpoint path when launching the script `anim.py`:
- for the oriented isotropic mean-curvature flow: `python anim.py logs/ModelDR/oriented_lp2_k17_zeros_s0`
- for the non-oriented isotropic mean-curvature flow: `python anim.py logs/ResidualParallel/nonoriented_lp2_k17_zeros_s1`
- for the non-oriented anisotropic mean-curvature flow: `python anim.py logs/ResidualParallel/nonoriented_lp4_k17_zeros_s1`

**Note** that the script must be launched from the folder where the model lies (commonly where the `logs` folder is) so that the loading process can find the model's file and class.

If you want to use a GPU for the inference, add the `--gpu` options.
You can also specify the domain's bounds with the `--bounds` option:
```bash
python anim.py --bounds [-1,1]x[-1,1] logs/ResidualParallel/nonoriented_lp4_k17_zeros_s1
```
and start paused with `--display_step 0`.

## Usage
Once launched, you can **draw** a phase with the left click and **erase** with the right click.

You can add an **inclusion disk** (there will be always a phase at that positon) at the cursor position with the `d` key, change its radius using the left click, move it using the middle click and delete it with the `Suppr` key.

The same way, you can add an **exclusion disk** (there will be never be a phase at that positon) with the `D` key, an **inclusion/exclusion circle** (`c` and `C` keys) or an **inclusion/exclusion segment** (`t` and `T` keys, use the left click to validat the end point).

You can also add **passive particles** that will (try) to follow the flow of the animation. It can be added at current position with `p` key, or initialy projected to the nearest interface with the `P` key.

**Animation speed** can be modified through the number of iterations per frame that is initialy set at 1 but can be decreased with the `-` key (0 means that the animation is paused) and increased with the `+` key.

Additionally, some **informations** may be displayed on the figure with the `i` key (otherwise, they are only displayed in the terminal).

And finally, you can **record** the animation using the `r` key to start and stop recording.

Enjoy!

## Demo

[nnpf_paint2d_record.webm](https://user-images.githubusercontent.com/10435058/207280773-535f708e-f6cd-456d-94e5-e3b71e6ea37e.webm)


