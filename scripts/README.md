# scripts/

Thin CLI entrypoints only: parse arguments, load a config (`promptcorrect.config`), call into `src/promptcorrect/`. No logic here.

Run on GreatLakes inside the container, e.g.:

```bash
apptainer exec --nv --bind /scratch/<ACCOUNT>_root/<ACCOUNT> --env PYTHONPATH=$PWD/src \
  $SIF python scripts/<script>.py --config configs/experiments/<x>.yaml
```
