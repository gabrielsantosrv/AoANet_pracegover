export PYTHONIOENCODING=utf8
export PYTHONPATH=/work/recod/gabriel.santos/AoANet

#python scripts/prepro_labels.py --input_json /work/recod/gabriel.santos/pracegover_dataset_400k/pracegover_dataset.json --output_json data/pracegover_400ktalk.json --output_h5 data/pracegover_400ktalk --max_length 100

python scripts/prepro_ngrams.py --input_json /work/recod/gabriel.santos/AoANet/data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train

#preprocessing image features - resnet101
#python scripts/prepro_feats.py --input_json /work/recod/gabriel.santos/pracegover_dataset_400k/pracegover_dataset.json --output_dir data/pracegovertalk_400k --images_root /work/recod/gabriel.santos/pracegover_dataset_400k


