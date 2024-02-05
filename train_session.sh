python train.py --experiment=f2b-pon --dataset=front2bev-aug --map_config=aug --model=pyramid --optimizer=adam --weight_mode=equal --pc=aircv 
python train.py --experiment=f2b-pon --dataset=front2bev-aug --map_config=aug --model=pyramid --optimizer=adam --weight_mode=inverse --pc=aircv 
python train.py --experiment=f2b-pon --dataset=front2bev-aug --map_config=aug --model=pyramid --optimizer=adam --weight_mode=sqrt_inverse --pc=aircv 
python train.py --experiment=f2b-pon --dataset=front2bev-aug --map_config=aug --model=pyramid --optimizer=adam --weight_mode=recall --pc=aircv 


python train.py --experiment=f2b-pon --dataset=front2bev --map_config=traffic --model=pyramid --optimizer=adam --weight_mode=equal --pc=aircv 
python train.py --experiment=f2b-pon --dataset=front2bev --map_config=traffic --model=pyramid --optimizer=adam --weight_mode=inverse --pc=aircv 
python train.py --experiment=f2b-pon --dataset=front2bev --map_config=traffic --model=pyramid --optimizer=adam --weight_mode=sqrt_inverse --pc=aircv 
python train.py --experiment=f2b-pon --dataset=front2bev --map_config=traffic --model=pyramid --optimizer=adam --weight_mode=recall --pc=aircv 



python train.py --experiment=f2b-ved --dataset=front2bev-aug --map_config=aug --model=ved --optimizer=adam --weight_mode=equal --pc=aircv 
python train.py --experiment=f2b-ved --dataset=front2bev-aug --map_config=aug --model=ved --optimizer=adam --weight_mode=inverse --pc=aircv 
python train.py --experiment=f2b-ved --dataset=front2bev-aug --map_config=aug --model=ved --optimizer=adam --weight_mode=sqrt_inverse --pc=aircv 
python train.py --experiment=f2b-ved --dataset=front2bev-aug --map_config=aug --model=ved --optimizer=adam --weight_mode=recall --pc=aircv 


python train.py --experiment=f2b-ved --dataset=front2bev --map_config=traffic --model=ved --optimizer=adam --weight_mode=equal --pc=aircv 
python train.py --experiment=f2b-ved --dataset=front2bev --map_config=traffic --model=ved --optimizer=adam --weight_mode=inverse --pc=aircv 
python train.py --experiment=f2b-ved --dataset=front2bev --map_config=traffic --model=ved --optimizer=adam --weight_mode=sqrt_inverse --pc=aircv 
python train.py --experiment=f2b-ved --dataset=front2bev --map_config=traffic --model=ved --optimizer=adam --weight_mode=recall --pc=aircv 

