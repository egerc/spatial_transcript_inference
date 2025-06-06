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
```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

mamba
```bash
mamba create -n myenv python=3.11
mamba activate myenv
pip install -r requirements.txt
```
