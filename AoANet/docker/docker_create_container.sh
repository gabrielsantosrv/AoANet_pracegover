#nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=0,1 --name gabriel_santos_container --userns=host -v /work/gabriel.santos:/work/gabriel.santos aoa_image

nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=6,7 --name gabriel_santos_container_6_7 --userns=host -v /work/recod/gabriel.santos:/work/recod/gabriel.santos gabrielsantosrv/attention_on_attention
