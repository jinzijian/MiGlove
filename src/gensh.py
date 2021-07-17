def gensh(lr):
    for i in range(len(lr)):
        print('/root/anaconda3/envs/new_env/bin/python3.6 main.py --gpu 0 --mimethod mine --method nmp --task eval --batch_size 2048 --repeat 1 --nonlinear elu --hidden_size 128 --lr '+ str(lr[i])' --milr 1e-6 --epoch 1000 --mode sample')