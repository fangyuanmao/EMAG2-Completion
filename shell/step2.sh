python scripts/generate_local_mask.py \
    --csv_path ./EMAG2/EMAG2_V3_values.csv \
    --output_path ./EMAG2/local_masks \
    --mask_size 10 &&

python repaint/test.py --conf_path repaint/confs/emag2_completion_local.yml &&

mkdir ./EMAG2/local_inpainted &&
cp -r ./log/emag2_local/inpainted ./EMAG2/local_inpainted/img &&
cp -r ./EMAG2/local_masks/recover_mask ./EMAG2/local_inpainted/mask &&
cp ./EMAG2/local_masks/centers.txt ./EMAG2/local_inpainted/centers.txt &&

python scripts/merge.py --EMAG2_V3_path ./EMAG2/EMAG2_V3_values_global_completion/emag2_complete.csv \
    --inpainted_path ./EMAG2/local_inpainted \
    --output_path ./EMAG2/EMAG2_V3_values_final_completion \
    --local
