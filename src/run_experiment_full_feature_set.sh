grans=("1d")  #  "1h" "2h" "4h" "6h" "8h" "12h" "30min"
m1s=("Kronos")  #  "Fincast" "Chronos2" "Tirex"
m2s=("randforest")  #  "TabPFN" "TabICL" "AutoGluon"
directions=("up")  #  "down"
output_root="/Volumes/Data/other/2026_NII/Output"  # TODO please adjust before running

for direction in "${directions[@]}"; do
  for m1 in "${m1s[@]}"; do
    for m2 in "${m2s[@]}"; do
      for gran in "${grans[@]}"; do
        echo "Running experiment for m1=$m1, m2=$m2, direction=$direction, gran=$gran"
        python experiment_full_feature_set.py \
                --output_root $output_root \
                --direction $direction \
                --m1 $m1 \
                --m2 $m2 \
                --gran $gran
      done
    done
  done
done