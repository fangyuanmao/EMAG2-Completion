python repaint/test.py --conf_path repaint/confs/emag2_completion_global.yml &&

python scripts/merge.py --EMAG2_V3_path ./EMAG2/EMAG2_V3_values.csv \
    --inpainted_path ./log/emag2_global/inpainted \
    --output_path ./EMAG2/EMAG2_V3_values_global_completion
