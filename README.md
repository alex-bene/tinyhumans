# TinyHumans
The goal of this repo (in the far future) is to provide a simple, lightweight, and easy-to-use library for working with virtual humans in PyTorch as existing libraries/repositories were either not maintained, too complicated or too sparse in their features.

For now, this repo is just a playground for me to experiment with different ideas and features. It is not close to being a library just yet.

## Mesh representations and rendering
- Currently, the library utilises `pytorch3d` and `trimesh` for mesh representations and `pyrender` for rendering.
- `trimesh` meshes and `pyrender` are used for rendering and visualization until I check if I can do this with `pytorch3d` in a performant and beautiful way.

## Environment setup
For now, this is only tested and developed using `uv` and an M1 Mac. I have no idea if it will work with other platforms or using other package managers to set up the environment.

After installing `uv` (can be achieved with `brew install uv in mac`) run the following commands:
```bash
uv python install 3.11
uv sync
```

If you also want to run the notebook demo, use:
```bash
uv sync --group demo
```

## TODOs
- [x] Add meshes rendering tools
- [x] Add SMPL-family body models
- [x] Fix bug where running `.to(...)`, `.clone()` and everything else copy-related for a `TensorDict` subclass, returns a `TensorDict` object instead of the original subclass. (tracking issue: https://github.com/pytorch/tensordict/issues/1184)
- [x] Add comments and docstrings
- [ ] Fix bug in tensordict where `bool` non-tensor data are transformed to 0-dim tensors, which prevents setting the batch size (tracking issue: https://github.com/pytorch/tensordict/issues/1199)
- [ ] Test SMPL-family body models
    - [x] tiny types tests
- [ ] Benchmark the difference of using `contiguous` tensors (e.g. repeat instead of expand) vs `non-contiguous`. Currently, we use expand in all `LimitedAttrTensorDictWithDefaults` objects for the default values, and also in `BaseParametricModel` forward to repeat class variables in each batch. In general, operations with `contiguous` tensors are faster (and from some isolated test, can be more than 2x faster or not faster at all). However, I need to fully this with the usual tensor dimensions used in this library to see whether I should bother at all.
- [ ] Add support for pointcloud registration
    - [ ] rigid alignment
    - [ ] non-rigid alignment

## Disclaimer (AI Stuff)
So, just a disclaimer, I've been playing around with [cline](https://github.com/cline/cline) (seriously, check it out if you haven't already). I've been using it with Gemini 2.0 Flash (free API and fast, yay) and Llama 3.3 70b from [Groq](https://groq.com/) (also free API and double fast, double yay). Writing comments and docstrings this way feels like cheating (in a good way), so yeah, most of those are probably AI-written with lite reviewing from me. Plus, I've been letting it draft some tests to start with. As I get more confident that it works well *enough* and as the library's documentation gets better, hopefully I'll be doing less and less coding myself.
