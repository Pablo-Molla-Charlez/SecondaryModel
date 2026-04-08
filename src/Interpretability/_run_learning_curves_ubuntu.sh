grans=("15m" "30m")   # "1d" "1h" "2h" "4h" "6h" "8h" "12h"
m1s=("Kronos" "Fincast")
directions=("up" "down")

for gran in "${grans[@]}"; do
  for m1 in "${m1s[@]}"; do
    for direction in "${directions[@]}"; do
      python learning_curves.py \
              --m1 $m1 \
              --gran $gran \
              --direction $direction \
              --output_root "/home/till/PycharmProjects/Secondary-Model/src/Output"
    done
  done
done

