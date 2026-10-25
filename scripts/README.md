# scripts/

Thin CLI entrypoints only: parse arguments, load a config (`promptcorrect.config`), call into `src/promptcorrect/`. No logic here.

Run on GreatLakes in the shared environment (`docs/environment.md`), e.g.:

```bash
source $PC_ROOT/venv/bin/activate
export PYTHONPATH=$PWD/src PC_PATHS=greatlakes
python scripts/<script>.py --config configs/experiments/<x>.yaml
```
