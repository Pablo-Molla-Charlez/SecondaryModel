grans=("1d" "1h" "2h" "4h" "6h" "8h" "12h" "15min" "30min")
m1s=("Kronos" "Fincast")

for gran in "${grans[@]}"; do
    for m1 in "${m1s[@]}"; do
        python learning_curves.py \
              --model1 $m1 \
              --granularity $gran
    done
done