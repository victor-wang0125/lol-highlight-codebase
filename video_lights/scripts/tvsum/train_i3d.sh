dset_name=tvsum
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results/tvsum/I3D_only
exp_id=exp-I3D-bicmf_2-csl-cal_0.2-nfr-edl_3-conval-hl


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/tvsum

# # video features
v_feat_dim=2048
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/video_features/)

# # text features
t_feat_dirs=()
t_feat_dirs+=(${feat_root}/query_features/)
t_feat_dim=512

#### training
bsz=4
lr=1e-3
dec_layers=3
enc_layers=3
bicmf_layers=2
hard_pos_neg_loss_coef=1
contrastive_align_loss_coef=0.2

######## TVSUM domain name
for dset_domain in BK BT DS FM GA MS PK PR VT VU
do
    ######## seeds
    for seed in 2018 #2429 2026 1009 2017
    do
        PYTHONPATH=$PYTHONPATH:. python video_lights/train.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dirs ${t_feat_dirs} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}/${seed}/${dset_domain} \
        --exp_id ${exp_id} \
        --hard_pos_neg_loss \
        --hard_pos_neg_loss_coef ${hard_pos_neg_loss_coef} \
        --contrastive_align_loss \
        --contrastive_align_loss_coef ${contrastive_align_loss_coef} \
        --dec_layers ${dec_layers} \
        --enc_layers ${enc_layers} \
        --bicmf_layers ${bicmf_layers} \
        --max_v_l 1000 \
        --n_epoch 2000 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --seed ${seed} \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        ${@:1}
    done
done
