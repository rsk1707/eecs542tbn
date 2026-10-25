# scripts/

Thin CLI entrypoints only: parse arguments, load a config (`promptcorrect.config`), call into `src/promptcorrect/`. No logic here.

Run on GreatLakes inside the container, e.g.:

```bash
singularity exec --nv --bind /scratch/eecs542f26s001_class_root/eecs542f26s001_class --env PYTHONPATH=$PWD/src \
  $SIF python scripts/<script>.py --config configs/experiments/<x>.yaml
```
