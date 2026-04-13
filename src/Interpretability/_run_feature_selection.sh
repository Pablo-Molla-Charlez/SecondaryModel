grans=("1d")   # "1d" "1h" "2h" "4h" "6h" "8h" "12h" "15m" "30m"
m1s=("Kronos" "Fincast")
directions=("up" "down")

for gran in "${grans[@]}"; do
  for m1 in "${m1s[@]}"; do
    for direction in "${directions[@]}"; do
      python feature_selection.py \
              --output_root "/Volumes/Data/other/2026_NII/Output" \
              --m1 $m1 \
              --gran $gran \
              --direction $direction \
              --n_splits 3 \
              --min_features 1 \
              --max_features 33
    done
  done
done

