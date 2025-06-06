# spatial_transcript_inference
using single cell sequencing based transcriptomics data to infer transcript levels of non covered genes in imaging based spatial transcriptomics data

## Clone the repo with submodules
```bash
git clone --recurse-submodules https://github.com/egerc/spatial_transcript_inference.git
```
## Install Environment
via uv (preferred)
```bash
uv venv
source .venv/bin/activate
uv pip install .
```

pip
make sure python version >= 3.11
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps .
```

mamba
```bash
mamba create -n benchmarking-env python=3.11
mamba activate benchmarking-env
pip install -r requirements.txt
```
