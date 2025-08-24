dset_name=tvsum
ctx_mode=video_tef
#v_feat_types=i3d
#v_feat_types=blip
#v_feat_types=i3d_blip
v_feat_types=slowfast_clip
#v_feat_types=slowfast_clip_blip
t_feat_types=clip
#t_feat_types=blip
#t_feat_types=clip_blip
results_root=results/tvsum/ft-v-${v_feat_types}-t-${t_feat_types}
exp_id=exp-I3D-bicmf_2-cal_0.8-nfr-edl_3-conval-hl


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/tvsum

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"i3d"* ]]; then
  v_feat_dirs+=(${feat_root}/video_features)
  (( v_feat_dim += 2048 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
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
lr=1e-3
dec_layers=3
enc_layers=3
bicmf_layers=1
hard_pos_neg_loss_coef=1
contrastive_align_loss_coef=0.8
cos_sim_loss_coef=1
#pretrain_path=results/pretrain/hl-video_tef-exp-bicmf_1-en_3-dec_3-tcl-hl-scsl-cal_0.2-slowfast_clip_blip-2024_11_20_06_24_46/model_best.ckpt
pretrain_path=results/pretrain/hl-video_tef-exp-bicmf_1-en_3-dec_3-tcl-hl-scsl-cal_0.2-slowfast_clip-2024_11_20_06_27_27/model_best.ckpt


for hard_pos_neg_loss_coef in 10 # 1 10
do
  for cos_sim_loss_coef in 1 # 1 5 10
  do
     ######## TVSUM domain name
     for dset_domain in  MS PK PR VT VU #BK BT DS FM GA MS PK PR VT VU
     do
        PYTHONPATH=$PYTHONPATH:. python video_lights/train.py \
        --dset_name ${dset_name} \
        --resume ${pretrain_path} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dirs ${t_feat_dirs} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}/hl_${hard_pos_neg_loss_coef}/csl_${cos_sim_loss_coef}/${dset_domain} \
        --exp_id ft-bicmf_${bicmf_layers}-en_${enc_layers}-dec_${dec_layers}-tcl-hl-csl_${cos_sim_loss_coef}-cal_${contrastive_align_loss_coef}-${v_feat_types} \
        --hard_pos_neg_loss \
        --hard_pos_neg_loss_coef ${hard_pos_neg_loss_coef} \
        --dec_layers ${dec_layers} \
        --enc_layers ${enc_layers} \
        --bicmf_layers ${bicmf_layers} \
        --max_v_l 1000 \
        --n_epoch 2000 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        --contrastive_align_loss \
        --contrastive_align_loss_coef ${contrastive_align_loss_coef} \
        --cos_sim_loss_coef ${cos_sim_loss_coef} \
        --mr_to_hd_loss \
        ${@:1}
    done
  done
done

