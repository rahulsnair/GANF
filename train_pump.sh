for seed in {18..20}
do 
    python -u train_pump.py\
        --seed=${seed}\
        --name=GANF_pump_seed_${seed}
done


