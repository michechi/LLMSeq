#!/bin/bash
echo "=== Avvio coda simulazioni ==="
echo "Inizio: $(date)"

# Le tue simulazioni (modifica con i tuoi parametri)
simulations=(
    "python3 Finetuning_DC.py  --model_name "meta-llama/Llama-3.1-8B" --model_type "general" --prompt_type "compact_no_time_rnd" --gradient_accumulation_steps 1 --batch_size 4 --patience 3  --max_length 2048 --seed 4550 --when_counting_death "last_visit" --max_visits 4 --peft"
    "python3 Finetuning_DC.py  --model_name "meta-llama/Llama-3.1-8B" --model_type "general" --prompt_type "compact_narrative" --gradient_accumulation_steps 1 --batch_size 4 --patience 3  --max_length 2048 --seed 4550 --when_counting_death "last_visit" --max_visits 4 --peft"
    "python3 Finetuning_DC.py  --model_name "meta-llama/Llama-3.1-8B" --model_type "general" --prompt_type "last_info_prompt" --gradient_accumulation_steps 1 --batch_size 4 --patience 3  --max_length 2048 --seed 4550 --when_counting_death "last_visit" --max_visits 4 --peft"
)

for i in "${!simulations[@]}"; do
    sim="${simulations[$i]}"
    echo ""
    echo "=== Simulazione $((i+1))/${#simulations[@]} ==="
    echo "Comando: $sim"
    echo "Inizio: $(date)"
    
    $sim
    
    echo "Fine: $(date)"
    echo "========================="
done

echo ""
echo "=== Tutte le simulazioni completate! ==="
echo "Fine: $(date)"