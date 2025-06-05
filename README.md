# spatial_transcript_inference
using single cell sequencing based transcriptomics data to infer transcript levels of non covered genes in imaging based spatial transcriptomics data

# Clone the repo with submodules
```bash
git clone --recurse-submodules https://github.com/egerc/spatial_transcript_inference.git
```

```bash
uv venv && source .venv/bin/activate && uv pip install .
```
mamba/conda alternative (discouraged, failed on my end):
```bash
mamba env create -f environments/environment.yml
```

Then you can select the environment as your jupyter kernel.
The main benchmarking notebook is notebooks/benchmarking_v2.ipynb