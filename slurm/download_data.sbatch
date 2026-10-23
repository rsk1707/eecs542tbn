#!/bin/bash
#SBATCH --job-name=pc_download
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=pc_download-%j.out
#
# Downloads COCO 2017, LVIS v1 annotations and the SAM 2.1 checkpoint into the shared scratch dir.
# Idempotent: re-submitting skips finished steps and resumes partial downloads.
#
#   export PC_ROOT=/scratch/eecs542f26s001_class_root/eecs542f26s001_class/srikrish/eecs542tbn
#   sbatch --account=eecs542f26s001_class slurm/download_data.sbatch
#
# SAM 3 / Grounding DINO weights come from Hugging Face: see scripts/download_hf.py (needs HF_TOKEN).
# Verify the URLs below still work before relying on them (they are from memory of the official sites).
set -euo pipefail
: "${PC_ROOT:?set PC_ROOT to the shared eecs542tbn directory}"

DATA="$PC_ROOT/data"
CKPT="$PC_ROOT/ckpt"
mkdir -p "$DATA/coco" "$DATA/lvis" "$CKPT/sam2.1"

# fetch <url> <dir> <marker> [unzip]: download (resumable), optionally unzip, drop the zip, mark done.
fetch() {
  local url=$1 dir=$2 marker=$3 unzip_it=${4:-}
  if [[ -f "$dir/.done_$marker" ]]; then echo "skip $marker (done)"; return; fi
  echo "== $marker: $url"
  wget -c -nv -P "$dir" "$url"
  local file="$dir/$(basename "$url")"
  if [[ -n "$unzip_it" ]]; then unzip -q -n "$file" -d "$dir"; rm -f "$file"; fi
  touch "$dir/.done_$marker"
}

# Small and needed first (annotations, val images), then the big train set.
fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$DATA/coco" annotations unzip
fetch http://images.cocodataset.org/zips/val2017.zip "$DATA/coco" val2017 unzip
fetch https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip "$DATA/lvis" lvis_val unzip
fetch https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip "$DATA/lvis" lvis_train unzip
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "$CKPT/sam2.1" sam21_large
fetch http://images.cocodataset.org/zips/train2017.zip "$DATA/coco" train2017 unzip

# Sharing is done with per-user ACLs (setfacl, see CLAUDE.md section 8), not chmod: on the class
# account the Unix group is the whole class, so group permissions would expose the data to classmates.
echo "all downloads finished"
