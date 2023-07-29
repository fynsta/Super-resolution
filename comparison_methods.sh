#!/bin/bash

SRF=${1:-2}
DATASET=${2:-"Set14"}

declare -a types

if [ "$DATASET" == "Set14" ]; then
  types=("bicubic" "glasner" "Kim" "nearest" "ScSR" "SelfExSR" "SRCNN" "GPR_RBF" "GPR_matern32")
else
  types=("bicubic" "GPR_RBF" "GPR_exponential" "GPR_matern32" "GPR_matern52"  )
fi

get_image_path() {
  local image="$1"
  local type="$2"

  if [ "$DATASET" == "Set14" ]; then
    local path="Set14/image_SRF_${SRF}/img_${image}_SRF_${SRF}_${type}.png"
  else
    local path="Set14_smaller/${image}_${type}_${SRF}x.png"
  fi

  echo "$path"
}

calculate_ssim_psnr() {
  local image1="$1"
  local image2="$2"
  local crop="$3"

  local ssim=$(
    magick compare -metric SSIM -colorspace gray -crop "$crop" \
      "$image1" "$image2" null: 2>&1
  )
  local psnr=$(
    magick compare -metric PSNR -colorspace gray -crop "$crop" \
      "$image1" "$image2" null: 2>&1 \
    | awk '{print $1}'
  )

  echo "$ssim $psnr"
}

get_crop() {
  local image="$1"

  local size=$(magick identify -format "%wx%h" "$image")
  local height=$(echo $size | cut -d'x' -f1)
  local width=$(echo $size | cut -d'x' -f2)
  local new_height=$(expr $height - 2 \* $SRF)
  local new_width=$(expr $width - 2 \* $SRF)
  local crop="${new_width}x${new_height}+$SRF+$SRF"

  echo "$crop"
}

declare -a averages_ssim
declare -a averages_psnr

for type in "${types[@]}"; do
  echo "----------------------------------------"
  echo "Type $type"
  sum_ssim=0
  sum_psnr=0
  for i in $(seq -f %03g 14); do
    read image1 <<< $(get_image_path "$i" "HR")
    read image2 <<< $(get_image_path "$i" "$type")

    read crop <<< $(get_crop "$image1")
    read ssim psnr <<< $(calculate_ssim_psnr "$image1" "$image2" "$crop")
    
    sum_ssim=$(echo "$sum_ssim + $ssim" | bc)
    sum_psnr=$(echo "$sum_psnr + $psnr" | bc)
  done

  average_ssim=$(echo "scale=3; $sum_ssim / 14" | bc) 
  average_psnr=$(echo "scale=3; $sum_psnr / 14" | bc)

  echo "Average SSIM value: $average_ssim"
  echo "Average PSNR value: $average_psnr"

  averages_ssim+=("$average_ssim")
  averages_psnr+=("$average_psnr")
done

echo "----------------------------------------"
max_ssim=0
for i in "${!averages_ssim[@]}"; do
  if (( $(echo "${averages_ssim[$i]} > $max_ssim" | bc -l) )); then
    max_ssim=${averages_ssim[$i]}
    best_ssim=${types[$i]}
  fi
done

echo "Best method (SSIM): $best_ssim with $max_ssim"

max_psnr=0
for i in "${!averages_psnr[@]}"; do
  if (( $(echo "${averages_psnr[$i]} > $max_psnr" | bc -l) )); then
    max_psnr=${averages_psnr[$i]}
    best_psnr=${types[$i]}
  fi
done
echo "Best method (PSNR): $best_psnr with $max_psnr"

