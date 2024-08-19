# PON
#python train.py --experiment=f2b-pon-rgb --dataset=f2b-rgbd --model=pon --optimizer=adam --pc=msi
#python train.py --experiment=f2b-pon_depth-rgbd --dataset=f2b-rgbd --model=pon_depth --optimizer=adam --pc=msi

# VED
#python train.py --experiment=f2b-ved-rgb --dataset=f2b-rgbd --map_config=traffic --model=ved --optimizer=adam --pc=home
#python train.py --experiment=f2b-rgved-rgbd --dataset=f2b-rgbd --map_config=traffic --model=rgved --optimizer=adam --pc=home

#
python train.py --experiment=f2b-vedfusion-rgbd --dataset=f2b-rgbd --map_config=traffic --model=vedfusion --optimizer=adam --pc=home
#python train.py --experiment=f2b-ponfusion-rgbd --dataset=f2b-rgbd --map_config=traffic --model=ponfusion --optimizer=adam --pc=home
