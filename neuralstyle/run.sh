#alias pythonmy=/home/zhangxuesen/installed/python/bin/python
export CUDA_VISIBLE_DEVICES=0
python fast_neural_style2.py \
	--MODEL_PATH=examples/model_style_Lonewolf \
	--STYLE_WEIGHT=1e1 \
	--TV_WEIGHT=10e-4 \
	--TRANSFORM_WEIGHT=0 \
	--STYLE_LAYERS=all \
	--EPOCHS=500 \
	--TRAIN_IMAGES_PATH=/home/zhangxs/data/test_art \
	--STYLE_IMAGES_PATH=/home/zhangxs/data/test_art \
