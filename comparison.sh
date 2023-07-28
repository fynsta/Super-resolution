#!/bin/bash

SRF=${1:-2}
declare -a types=("bicubic" "glasner" "Kim" "nearest" "ScSR" "SelfExSR" "SRCNN" "GPR")

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

for type in "${types[@]}"; do
  sum_ssim=0
  sum_psnr=0
  for i in $(seq -f %03g 1 14); do
    image1="Set14/image_SRF_${SRF}/img_${i}_SRF_${SRF}_HR.png"
    image2="Set14/image_SRF_${SRF}/img_${i}_SRF_${SRF}_${type}.png"
    read crop <<< $(get_crop "$image1")
    read ssim psnr <<< $(calculate_ssim_psnr "$image1" "$image2" "$crop")
    
    sum_ssim=$(echo "$sum_ssim + $ssim" | bc)
    sum_psnr=$(echo "$sum_psnr + $psnr" | bc)
  done

  average_ssim=$(echo "scale=3; $sum_ssim / 14" | bc) 
  average_psnr=$(echo "scale=3; $sum_psnr / 14" | bc)

  echo "Average SSIM value: $average_ssim for $type"
  echo "Average PSNR value: $average_psnr for $type"
done