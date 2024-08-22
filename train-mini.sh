# VED  - RGB - AUG
#python train.py --experiment=f2b-mini-ved-rgb --dataset=f2b-mini-rgbd --model=ved --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-ved-rgb-aug_cl --dataset=f2b-mini-rgbd-aug_cl --model=ved --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-ved-rgb-lbda --dataset=f2b-mini-rgbd-lbda --model=ved --optimizer=adam --pc=aircv

# PON - RGB - AUG
#python train.py --experiment=f2b-mini-pon-rgb --dataset=f2b-mini-rgbd --model=pon --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-pon-rgb-aug_cl --dataset=f2b-mini-rgbd-aug_cl --model=pon --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-pon-rgb-lbda --dataset=f2b-mini-rgbd-lbda --model=pon --optimizer=adam --pc=aircv

# VED - RGBD - 4CH
#python train.py --experiment=f2b-mini-rgved-rgbd --dataset=f2b-mini-rgbd --model=rgved --optimizer=adam --pc=aircv
python train.py --experiment=f2b-mini-rgved-rgbd-aug_cl --dataset=f2b-mini-rgbd-aug_cl --model=rgved --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-rgved-rgbd-lbda --dataset=f2b-mini-rgbd-lbda --model=rgved --optimizer=adam --pc=aircv

# PON - RGBD - 4CH
#python train.py --experiment=f2b-mini-pon_depth-rgbd --dataset=f2b-mini-rgbd --model=pon_depth --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-pon_depth-rgbd-aug_cl --dataset=f2b-mini-rgbd-aug_cl --model=pon_depth --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-pon_depth-rgbd-lbda --dataset=f2b-mini-rgbd-lbda --model=pon_depth --optimizer=adam --pc=aircv

# VED - RGBD - FUSION
python train.py --experiment=f2b-mini-vedfusion-rgbd --dataset=f2b-mini-rgbd --model=ved_fusion --optimizer=adam --pc=aircv
python train.py --experiment=f2b-mini-vedfusion-rgbd-aug_cl --dataset=f2b-mini-rgbd-aug_cl --model=ved_fusion --optimizer=adam --pc=aircv
python train.py --experiment=f2b-mini-vedfusion-rgbd-lbda --dataset=f2b-mini-rgbd-lbda --model=ved_fusion --optimizer=adam --pc=aircv

# PON - RGBD - FUSION
#python train.py --experiment=f2b-mini-ponfusion-rgbd --dataset=f2b-mini-rgbd --model=ponfusion --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-ponfusion-rgbd-aug_cl --dataset=f2b-mini-rgbd-aug_cl --model=ponfusion --optimizer=adam --pc=aircv
#python train.py --experiment=f2b-mini-ponfusion-rgbd-lbda --dataset=f2b-mini-rgbd-lbda --model=ponfusion --optimizer=adam --pc=aircv
