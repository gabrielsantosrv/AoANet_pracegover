export PYTHONIOENCODING=utf8
export PYTHONPATH=/work/recod/gabriel.santos/AoANet

base=cider
log_dir=log_len_100_$base

python eval.py --dump_images 0\
 --dump_json 1\
 --num_images -1\
 --model $log_dir/log_aoanet_rl/model.pth\
 --infos_path $log_dir/log_aoanet_rl/infos_aoanet.pkl\
 --language_eval 1\
 --image_root /work/recod/gabriel.santos/pracegover_dataset_400k\
 --beam_size 1\
 --batch_size 100\
 --split val\
  --max_length 100\
 --output_file ./val_len_100_$base.json

