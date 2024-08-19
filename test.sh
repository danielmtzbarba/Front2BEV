python test.py --experiment=f2b-pon-rgb --dataset=f2b-rgbd --model=pon --optimizer=adam --pc=msi
python test.py --experiment=f2b-ved-rgb --dataset=f2b-rgbd --model=ved --optimizer=adam --pc=msi

python test.py --experiment=f2b-pon_depth-rgbd --dataset=f2b-rgbd --model=pon_depth --optimizer=adam --pc=msi
python test.py --experiment=f2b-rgved-rgbd --dataset=f2b-rgbd --model=rgved --optimizer=adam --pc=msi

python test.py --experiment=f2b-vedfusion-rgbd --dataset=f2b-rgbd --model=ved_fusion --optimizer=adam --pc=msi
