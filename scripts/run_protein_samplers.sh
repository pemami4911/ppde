proteins=("PABP_YEAST_Fields2013" "UBE4B_MOUSE_Klevit2013-nscor_log2_ratio" "GFP_AEQVI_Sarkisyan2016")
priors=("transformer")
SEED=100

for prot in "${proteins[@]}"
do

    for prior in "${priors[@]}"
    do   
        if [ $prior = "potts" ] && [ $prot = "PABP_YEAST_Fields2013" ]; then
            LAMDA=5
            MSA="PABP_YEAST.a2m"
        elif [ $prior = "potts" ] && [ $prot = "UBE4B_MOUSE_Klevit2013-nscor_log2_ratio" ]; then
            LAMDA=0.5
            MSA="UBE4B_MOUSE.a2m"
        elif [ $prior = "potts" ] && [ $prot = "GFP_AEQVI_Sarkisyan2016" ]; then
            LAMDA=15
            MSA="GFP_AEQVI.a2m"
        elif [ $prior = "transformer" ] && [ $prot = "PABP_YEAST_Fields2013" ]; then
            LAMDA=5
            MSA="PABP_YEAST.a2m"
        elif [ $prior = "transformer" ] && [ $prot = "UBE4B_MOUSE_Klevit2013-nscor_log2_ratio" ]; then
            LAMDA=3
            MSA="UBE4B_MOUSE.a2m"
        elif [ $prior = "transformer" ] && [ $prot = "GFP_AEQVI_Sarkisyan2016" ]; then
            LAMDA=1
            MSA="GFP_AEQVI.a2m"
        fi
        python3 scripts/directed_evolution.py --seed 1 --sampler PPDE --run_signature $prior --unsupervised_expert $prior --energy_function product_of_experts --energy_lamda $LAMDA --n_iters 10000 --log_every 100 --protein $prot --msa_path data/proteins/$MSA --nmut_threshold 10
    done
done

wait;