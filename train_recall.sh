python train.py --experiment=f2b --dataset=front2bev --map_config=traffic --model=pon --optimizer=sgd --weight_mode=recall --pc=aircv 
python train.py --experiment=f2b --dataset=front2bev-aug --map_config=aug --model=pon --optimizer=sgd --weight_mode=recall --pc=aircv 

python train.py --experiment=f2b --dataset=front2bev --map_config=traffic --model=ved --optimizer=sgd --weight_mode=recall --pc=aircv 
python train.py --experiment=f2b --dataset=front2bev-aug --map_config=aug --model=ved --optimizer=sgd --weight_mode=recall --pc=aircv 
