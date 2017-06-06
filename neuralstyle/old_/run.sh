alias python_c=/home/ldap/zhangxuesen/installed/python_cpu/bin/python
alias python_g=/home/ldap/zhangxuesen/installed/python_gpu/bin/python
python_c fast_neural_style.py \
	--MODEL=sli \
	--MODEL_PATH=examples/model_style5 \
        --TRAIN_IMAGES_PATH=/home/ldap/zhangxuesen/Data/try_coco \
	--STYLE_WEIGHT=1e1 \
	--TV_WEIGHT=1e-4 \
	--STYLE_LAYERS=relu1_2,relu2_2,relu3_3,relu4_3 \
#	--STYLE_IMAGES_PATH=/home/ldap/zhangxuesen/Data/coco \
#	--TRANSFORM_WEIGHT=1e0 \
