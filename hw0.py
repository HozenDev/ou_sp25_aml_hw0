import sys
import argparse
import copy
import pickle
import random
import numpy as np
import os
import wandb

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, InputLayer, Dense
from keras import Input, Model
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib import colors

def build_model(n_inputs:int, n_hidden:list, n_output:int, activation:str='relu', lrate:float=0.001)-> Sequential:
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,)))
    
    for i, n in enumerate(n_hidden):
        model.add(Dense(n, use_bias=True, activation=activation, name='hidden%d'%i))
    
    model.add(Dense(n_output, use_bias=True, activation=activation, name='output'))
    
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(loss='mse', optimizer=opt)
    
    return model

def execute_exp(args:argparse.ArgumentParser):
    with open("/home/fagg/datasets/hw0/hw0_dataset.pkl", "rb") as fp:
        dictionary = pickle.load(fp)
    ins = dictionary["ins"]
    outs = dictionary["outs"]
    
    model = build_model(ins.shape[1], args.hidden, outs.shape[1], activation='relu', lrate=args.lrate)
    
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=1000,
                                                      restore_best_weights=True,
                                                      min_delta=0.001,
                                                      monitor='loss')
    
    argstring = "exp_%02d_hidden_%s"%(args.exp, '_'.join([str(i) for i in args.hidden]))
    print("EXPERIMENT: %s"%argstring)
    fname_output = "results/hw0_results_%s.pkl"%(argstring)
    
    if os.path.exists(fname_output):
        print("File %s already exists."%fname_output)
        return
    
    wandb.init(project="hw0-deep-learning", name=argstring)
    
    if not args.nogo:
        history = model.fit(x=ins, y=outs,
                            epochs=args.epochs, batch_size=args.batch_size,
                            verbose=args.verbose>=2, callbacks=[early_stopping_cb])
        predictions = model.predict(ins)
        abs_errors = np.abs(predictions - outs)
        max_error = np.max(abs_errors)
        sum_error = np.sum(abs_errors)
        count_large_errors = np.sum(abs_errors > 0.1)
        
        wandb.log({"max_abs_error": max_error, "sum_abs_error": sum_error, "count_large_errors": count_large_errors})
        
        with open(fname_output, "wb") as fp:
            pickle.dump({"history": history.history, "errors": abs_errors}, fp)
            pickle.dump(args, fp)
        
        wandb.finish()

def plot_results():
    files = [f for f in os.listdir("results/") if f.startswith("hw0_results_") and f.endswith(".pkl")]
    plt.figure()
    for f in files:
        with open("results/" + f, "rb") as fp:
            data = pickle.load(fp)
            plt.plot(data["history"]["loss"], label=f)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/learning_curves.png")
    
    abs_errors = []
    for f in files:
        with open("results/" + f, "rb") as fp:
            data = pickle.load(fp)
            abs_errors.extend(data["errors"].flatten())
    plt.figure()
    plt.hist(abs_errors, bins=50)
    plt.ylim([0,50])
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.savefig("results/error_histogram.png")

def create_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Deep Learning Experiment')
    parser.add_argument('--exp', type=int, default=0, help='Experiment number')
    parser.add_argument('--lrate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden', nargs='+', type=int, default=[10,5], help='Hidden layer configuration')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    execute_exp(args)
    plot_results()
