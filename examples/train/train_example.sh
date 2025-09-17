# Set your HF_TOKEN as an environment variable or uncomment and set it here
# export HF_TOKEN="your_token_here"
# ensure Huggingface Hub uses the token
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/home/daksh/MTP/DeSTA2/DeSTA2.5-Audio":$PYTHONPATH
export ROOT_DIR="/home/daksh/MTP/DeSTA2/DeSTA2.5-Audio"


config=desta25_llama31-8B_Qformer6L.yaml
dataset_config=debug_local_libritts
devices=1

data_root=/home/daksh/data/DeSTA2Audio/LibriTTS_R

project=desta2  # Change this to your project name
name="indic_whisper_attempt"
exp_name=$(date +%y%m%d-%H)@${name}
exp_dir="${ROOT_DIR}/my_exps/${project}/${exp_name}"


resume_from_checkpoint=null
init_from_pretrained_weights=null


# record git diff
mkdir -p ${exp_dir}
git diff > ${exp_dir}/git-diff.txt
nvidia-smi > ${exp_dir}/nvidia-smi.txt

python ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path ${ROOT_DIR}/examples/train/config_trial \
    --config-name=${config} \
    trainer.devices=${devices} \
    +dataset=${dataset_config} \
    +exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    dataset.train_ds.data_root=${data_root} \
    dataset.validation_ds.data_root=${data_root} \
    +resume_from_checkpoint=${resume_from_checkpoint} \
    +init_from_pretrained_weights=${init_from_pretrained_weights}
