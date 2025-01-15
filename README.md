# TinyHumans
The goal of this repo (in the far future) is to provide a simple, lightweight, and easy-to-use library for working with virtual humans in PyTorch as existing libraries/repositories were either not maintained, too complicated or too sparse in their features.

For now, this repo is just a playground for me to experiment with different ideas and features. It is not anywhere near being a library.

## Thoughts
- Currently utilises `pytorch3d` and `trimesh` for mesh representations and `pyrender` for rendering.
- `trimesh` meshes and `pyrender` are used for rendering and visualization until I check if I can do this with `pytorch3d` in a performant and beautiful way.

## Environment setup
For now, this is only tested and developed using `uv` and an M1 Mac. I have no idea if it will work with other platforms or using other package managers to set up the environment.

After installing `uv` (can be achieved with `brew install uv in mac`) run the following commands:
```bash
uv python install 3.11
uv sync
uv pip install "git+https://github.com/nghorbani/human_body_prior" # to run vposer.ipynb
uv pip install "git+https://github.com/facebookresearch/pytorch3d" --no-build-isolation
```

## TODOs
- [x] Add meshes rendering tools
- [x] Add SMPL-family body models
- [ ] Fix bug where running `.to(...)`, `.clone()` and everything else copy-related for a `TensorDict` subclass, returns a `TensorDict` object instead of the original subclass.
- [ ] Test SMPL-family body models
- [ ] Benchmark the difference of using `contiguous` tensors (e.g. repeat instead of expand) vs `non-contiguous`. Currently, we use expand in all `LimitedAttrTensorDictWithDefaults` objects for the default values, and also in `BaseParametricModel` forward to repeat class variables in each batch. In general, operations with `contiguous` tensors are faster (and from some isolated test, can be more than 2x faster or not faster at all). However, I need to fully this with the usual tensor dimensions used in this library to see whether I should bother at all.
- [ ] Add comments and docstrings
- [ ] Add support for pointcloud registration
    - [ ] rigid alignment
    - [ ] non-rigid alignment
