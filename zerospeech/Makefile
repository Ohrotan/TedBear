cnt = 500000
preprocess:
	python preprocess.py in_dir=english dataset=english
train:
	python train.py checkpoint_dir=./ dataset=english
convert:
	python convert.py checkpoint=checkpoints/model.ckpt-$(cnt).pt in_dir=./ out_dir=submission/english/test_ synthesis_list=datasets/english/synthesis.json dataset=english
