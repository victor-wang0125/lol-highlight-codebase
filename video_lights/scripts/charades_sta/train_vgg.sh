dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=vgg
t_feat_types=clip
results_root=results/charadesSTA
exp_id=exp

######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/processed/charades

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"vgg"* ]]; then
  v_feat_dirs+=(${feat_root}/vgg_features/rgb_features)
  (( v_feat_dim += 4096 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi


# text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/clip_text_features)
  (( t_feat_dim += 300 ))  # for vgg we shall use glove embeddings
fi

echo $t_feat_dim
echo ${t_feat_dirs[@]}

#### training
bsz=8
eval_bsz=16
hidden_dim=256
enc_layers=3
dec_layers=3

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
--results_root ${results_root} \
--exp_id ${exp_id} \
--max_v_l -1 \
--clip_length 0.166666 \
--lr 0.0002 \
--n_epoch 100 \
--contrastive_align_loss \
--mr_to_hd_loss \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 4 \
--dropout 0.2 \
--hidden_dim ${hidden_dim} \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--eval_bsz ${eval_bsz} \
${@:1}
