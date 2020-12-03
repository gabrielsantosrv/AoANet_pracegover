export PYTHONIOENCODING=utf8
export PYTHONPATH=/work/recod/gabriel.santos/bert_AoANet/AoANet

python eval.py --dump_images 0 --dump_json 1 --num_images -1 --model log_bertscore/log_aoanet/model-best.pth --infos_path log_bertscore/log_aoanet/infos_aoanet-best.pkl --language_eval 1 --image_root /work/recod/gabriel.santos/pracegover_dataset --beam_size 1 --batch_size 100 --split val  --max_length 100

