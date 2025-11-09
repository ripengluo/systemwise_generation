#num_sample=`ls out_new/ | grep sample* |wc -l` 
#mv out_new/sample_000 out_new/sample_`printf "%03d" "$num_sample"`
python gen_crystal.py --site-ckpt models/pretrain_last.pt --fr-sites fr_sites.dat --chgcar trigonal/sample_052/CHGCAR 
