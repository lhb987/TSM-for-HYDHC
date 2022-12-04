python main.py DHC RGB \
     --arch resnet50 --num_segments 80 \
     --gd 20 --lr 0.002 --lr_steps 8 13 --epochs 15 \
     --batch-size 8 -j 16 --dropout 0.4 --consensus_type=avg --eval-freq=1 \
     --dct --shift --shift_div=8 --shift_place=blockres --npb
