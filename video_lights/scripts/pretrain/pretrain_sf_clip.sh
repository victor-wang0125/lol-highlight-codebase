dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_types=clip
results_root=results/pretrain/final
exp_id=pt_sf-clip_fuse_calign_cmf

######## data paths
train_path=data/pretrain/pre_train_blip.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../pretrain/pretrain_features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"blip"* ]]; then
  v_feat_dirs+=(${feat_root}/blip_video_features)
  (( v_feat_dim += 768 ))
fi

# text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/clip_query_features)
  (( t_feat_dim += 512 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${t_feat_types} == *"blip"* ]]; then
  t_feat_dirs+=(${feat_root}/blip_query_features)
  (( t_feat_dim += 768 ))
fi

#echo "t_feat_dirs: "$t_feat_dirs
#### training
bsz=256
num_workers=8
n_epoch=50
max_es_cnt=100
max_v_l=75
dec_layers=3
enc_layers=3
bicmf_layers=1
contrastive_align_loss_coef=0.01


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
--num_workers ${num_workers} \
--max_v_l ${max_v_l} \
--n_epoch ${n_epoch} \
--max_es_cnt ${max_es_cnt} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
--bicmf_layers ${bicmf_layers} \
--mr_to_hd_loss \
--hard_pos_neg_loss \
--contrastive_align_loss \
--contrastive_align_loss_coef ${contrastive_align_loss_coef} \
--results_root ${results_root} \
--exp_id exp-bicmf_${bicmf_layers}-en_${enc_layers}-dec_${dec_layers}-tcl-hl-scsl-cal_${contrastive_align_loss_coef}-${v_feat_types} \
--device 0 \
--hidden_dim 256 \
${@:1}

