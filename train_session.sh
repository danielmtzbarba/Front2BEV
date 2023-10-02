echo 'Training 3k (no weights)'
python train.py

echo 'Training 3k (class weights)'

python train_w.py

echo 'Training 3k (fov class weights)'

python train_fov.py

