# TinyHumans
The goal of this repo (in the far future) is to provide a simple, lightweight, and easy-to-use library for working with virtual humans in PyTorch as existing libraries/repositories were either not maintained, too complicated or too sparse in their features.

For now, this repo is just a playground for me to experiment with different ideas and features. It is not anywhere near being a library.

## Thoughts
- Currently utilises `pytorch3d` and `trimesh` for mesh representations, `pyrender` for rendering and `human_body_prior` for SMPL-family body models.
- I think, I want to mostly support `pytorch3d` meshes as this would be the most straightforward for training with `pytorch`.
- `trimesh` meshes and `pyrender` will be used for rendering and visualization if I can't find a way to do it with `pytorch3d` in a performant and beautiful way.

## Environment setup
For now, this is only tested and developed with `uv` in arm macs. I have no idea if it will work with other platforms or using other package managers to set up the environment.

After installing `uv` (can be achieved with `brew install uv in mac`) run the following commands:
```bash
uv python install 3.11
uv sync
uv pip install "git+https://github.com/nghorbani/human_body_prior"
uv pip install "git+https://github.com/facebookresearch/pytorch3d" --no-build-isolation
```
