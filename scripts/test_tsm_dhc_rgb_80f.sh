python main.py DHC RGB \
     --evaluate --resume 'checkpoint/ckpt.best.pth.tar' \
     --arch resnet50 --num_segments 80 \
     --gd 20 --lr 0.00125 --lr_steps 10 20 --epochs 25 \
     --batch-size 8 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb