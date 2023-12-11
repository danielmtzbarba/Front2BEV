echo 'Training all_layers 2k \n'
python train.py -c=layers_all -k=2 -e=10

echo 'Training all_layers 3k \n'
python train.py -c=layers_all -k=3 -e=10

echo 'Training layers_none 2k \n'
python train.py -c=layers_none -k=2 -e=10

echo 'Training layers_none 3k \n'
python train.py -c=layers_none -k=3 -e=10

echo 'Training traffic 2k \n'
python train.py -c=traffic -k=2 -e=10

echo 'Training traffic 3k \n'
python train.py -c=traffic -k=3 -e=10

echo 'Training traffic 4k \n'
python train.py -c=traffic -k=4 -e=10

echo 'Training traffic 5k \n'
python train.py -c=traffic -k=5 -e=10

echo 'Training traffic 6k \n'
python train.py -c=traffic -k=6 -e=10