#!/bin/bash

types=("bicubic" "GPR") # ("nearest" "ScSR" "SelfExSR" "SRCNN" "glasner" "Kim")
SRF="2"

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

for i in $(seq -f %03g 0 14); do
  echo "----------------------------------------"
  echo "Image $i"
  for type in "${types[@]}"; do
    image1="Set14/image_SRF_${SRF}/img_${i}_SRF_${SRF}_HR.png"
    image2="Set14/image_SRF_${SRF}/img_${i}_SRF_${SRF}_${type}.png"
    read crop <<< $(get_crop "$image1")
    read ssim psnr <<< $(calculate_ssim_psnr "$image1" "$image2" "$crop")
    
    echo $type "SSIM:" $ssim "PSNR:" $psnr
  done
done