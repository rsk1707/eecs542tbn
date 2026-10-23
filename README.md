# eecs542tbn

Code for the UMich EECS 542 (Advanced Topics in Computer Vision) project:
**Learning to Correct Prompts for Frozen Segmentation Foundation Models.**

A small corrector network looks at a frozen segmenter's (SAM 3) current mask and prompt, and proposes corrective clicks or box adjustments. The segmenter is never fine-tuned.

**Start here:** [`CLAUDE.md`](CLAUDE.md) (project context, roles, conventions), [`docs/interfaces.md`](docs/interfaces.md) (shared formats), [`docs/demo_sprint.md`](docs/demo_sprint.md) (current sprint plan), [`docs/container.md`](docs/container.md) (environment on GreatLakes).

## Local setup (laptop, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

All development and unit tests run against `DummySegmenter` (no GPU, no data download). Real runs happen on GreatLakes inside the Apptainer image.

## Layout

```
configs/{paths,experiments}/   path roots per machine; run configs
container/                     Apptainer definition
docs/                          interfaces, sprint plan, container build
slurm/                         sbatch templates
scripts/                       thin CLI entrypoints (logic lives in src/)
src/promptcorrect/             types, config, data, segmenters, detectors, perturb, baselines, eval, corrector, viz
tests/                         CPU tests on the dummy segmenter
```
