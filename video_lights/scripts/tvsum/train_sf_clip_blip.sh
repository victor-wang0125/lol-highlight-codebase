dset_name=tvsum
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_types=clip
results_root=results/tvsum
exp_id=sf-clip-blip_calign_1e-5


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/processed/tvsum

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/sf_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"blip"* ]]; then
  v_feat_dirs+=(${feat_root}/video_features_blip)
  (( v_feat_dim += 768 ))
fi

echo $v_feat_dim
echo ${v_feat_dirs[@]}

# text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/query_features)
  (( t_feat_dim += 512 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${t_feat_types} == *"blip"* ]]; then
  t_feat_dirs+=(${feat_root}/query_features_blip)
  (( t_feat_dim += 768 ))
fi

echo $t_feat_dim
echo ${t_feat_dirs[@]}

#### training
bsz=4
lr=1e-4

######## TVSUM domain name
for dset_domain in BK BT DS FM GA MS PK PR VT VU #
do
    ######## seeds
    for seed in  0 1 2 2017 2018 #
    do
        PYTHONPATH=$PYTHONPATH:. python video_lights/train.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dirs ${t_feat_dirs[@]} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}/${seed}/${dset_domain} \
        --exp_id ${exp_id}_$seed \
        --contrastive_align_loss \
        --max_v_l 1000 \
        --n_epoch 3000 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --seed $seed \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        ${@:1}
    done
done
