#!/bin/bash

SRF=${1:-2}
DATASET=${2:-"Set14"}

types=("bicubic" "GPR_RBF" "GPR_exponential" "GPR_matern32" ) # ("nearest" "ScSR" "SelfExSR" "SRCNN" "glasner" "Kim")

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

for i in $(seq -f %03g 14); do
  echo "----------------------------------------"
  echo "Image $i"
  for type in "${types[@]}"; do
    read image1 <<< $(get_image_path "$i" "HR")
    read image2 <<< $(get_image_path "$i" "$type")

    read crop <<< $(get_crop "$image1")
    read ssim psnr <<< $(calculate_ssim_psnr "$image1" "$image2" "$crop")
    
    echo $type "SSIM:" $ssim "PSNR:" $psnr
  done
done