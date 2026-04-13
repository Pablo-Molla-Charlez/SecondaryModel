grans=("1d")   # "1h" "2h" "4h" "6h" "8h" "12h"
m1s=("Kronos")  #  "Fincast"
directions=("up")  #  "down"

for gran in "${grans[@]}"; do
  for m1 in "${m1s[@]}"; do
    for direction in "${directions[@]}"; do
      python feature_selection.py \
              --output_root "/home/till/PycharmProjects/Secondary-Model/src/Output" \
              --n_splits 10 \
              --gran $gran \
              --m1 $m1 \
              --direction $direction \
              --min_features 1 \
              --max_features 33
    done
  done
done

