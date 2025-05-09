#!/usr/bin/env python3

import itertools
import argparse
import ctypes as c
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import Pool, cpu_count
from scipy.stats import norm
import random

# For saving models
import pickle
import lzma

from wisard import WiSARD

# For the tabular datasets (all except MNIST)
import tabular_tools

POPULATION_SIZE = 20
GENERATIONS = 15
MUTATION_RATE = 0.25


# Perform inference operations using provided test set on provided model with specified bleaching value (default 1)
def run_inference(inputs, labels, model, bleach=1):
    num_samples = len(inputs)
    correct = 0
    ties = 0
    model.set_bleaching(bleach)
    for d in range(num_samples):
        prediction = model.predict(inputs[d])
        label = labels[d]
        if len(prediction) > 1:
            ties += 1
        if prediction[0] == label:
            correct += 1
    correct_percent = round((100 * correct) / num_samples, 4)
    tie_percent = round((100 * ties) / num_samples, 4)
    # print(f"With bleaching={bleach}, accuracy={correct}/{num_samples} ({correct_percent}%); ties={ties}/{num_samples} ({tie_percent}%)")
    return correct

def parameterized_run(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, unit_inputs, unit_entries, unit_hashes):
    model = WiSARD(train_inputs[0].size, train_labels.max()+1, unit_inputs, unit_entries, unit_hashes)

    # print("Training model")
    for d in range(len(train_inputs)):
        model.train(train_inputs[d], train_labels[d])
        # if ((d+1) % 10000) == 0:
        #     print(d+1)

    max_val = 0
    for d in model.discriminators:
        for f in d.filters:
            max_val = max(max_val, f.data.max())
    # print(f"Maximum possible bleach value is {max_val}")
    # Use a binary search-based strategy to find the value of b that maximizes accuracy on the validation set
    best_bleach = max_val // 2
    step = max(max_val // 4, 1)
    bleach_accuracies = {}
    while True:
        values = [best_bleach-step, best_bleach, best_bleach+step]
        accuracies = []
        for b in values:
            if b in bleach_accuracies:
                accuracies.append(bleach_accuracies[b])
            elif b < 1:
                accuracies.append(0)
            else:
                accuracy = run_inference(val_inputs, val_labels, model, b)
                bleach_accuracies[b] = accuracy
                accuracies.append(accuracy)
        new_best_bleach = values[accuracies.index(max(accuracies))]
        if (new_best_bleach == best_bleach) and (step == 1):
            break
        best_bleach = new_best_bleach
        if step > 1:
            step //= 2
    # print(f"Best bleach: {best_bleach}; inputs/entries/hashes = {unit_inputs},{unit_entries},{unit_hashes}")
    # Evaluate on test set
    # print("Testing model")
    accuracy = run_inference(test_inputs, test_labels, model, bleach=best_bleach)
    return model, accuracy

# Convert input dataset to binary representation
# Use a thermometer encoding with a configurable number of bits per input
# A thermometer encoding is a binary encoding in which subsequent bits are set as the value increases
#  e.g. 0000 => 0001 => 0011 => 0111 => 1111
def binarize_datasets(train_dataset, test_dataset, bits_per_input, separate_validation_dset=None, train_val_split_ratio=0.9):
    # Given a Gaussian with mean=0 and std=1, choose values which divide the distribution into regions of equal probability
    # This will be used to determine thresholds for the thermometer encoding
    std_skews = [norm.ppf((i+1)/(bits_per_input+1))
                 for i in range(bits_per_input)]

    print("Binarizing train/validation dataset")
    train_inputs = []
    train_labels = []
    for d in train_dataset:
        # Expects inputs to be already flattened numpy arrays
        train_inputs.append(d[0])
        train_labels.append(d[1])
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    use_gaussian_encoding = True
    if use_gaussian_encoding:
        mean_inputs = train_inputs.mean(axis=0)
        std_inputs = train_inputs.std(axis=0)
        train_binarizations = []
        for i in std_skews:
            train_binarizations.append(
                (train_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        min_inputs = train_inputs.min(axis=0)
        max_inputs = train_inputs.max(axis=0)
        train_binarizations = []
        for i in range(bits_per_input):
            train_binarizations.append(
                (train_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))

    # Creates thermometer encoding
    train_inputs = np.concatenate(train_binarizations, axis=1)

    # Ideally, we would perform bleaching using a separate dataset from the training set
    #  (we call this the "validation set", though this is arguably a misnomer),
    #  since much of the point of bleaching is to improve generalization to new data.
    # However, some of the datasets we use are simply too small for this to be effective;
    #  a very small bleaching/validation set essentially fits to random noise,
    #  and making the set larger decreases the size of the training set too much.
    # In these cases, we use the same dataset for training and validation
    if separate_validation_dset is None:
        separate_validation_dset = (len(train_inputs) > 10000)
    if separate_validation_dset:
        split = int(train_val_split_ratio*len(train_inputs))
        val_inputs = train_inputs[split:]
        val_labels = train_labels[split:]
        train_inputs = train_inputs[:split]
        train_labels = train_labels[:split]
    else:
        val_inputs = train_inputs
        val_labels = train_labels

    print("Binarizing test dataset")
    test_inputs = []
    test_labels = []
    for d in test_dataset:
        # Expects inputs to be already flattened numpy arrays
        test_inputs.append(d[0])
        test_labels.append(d[1])
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)
    test_binarizations = []
    if use_gaussian_encoding:
        for i in std_skews:
            test_binarizations.append(
                (test_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        for i in range(bits_per_input):
            test_binarizations.append(
                (test_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))
    test_inputs = np.concatenate(test_binarizations, axis=1)

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels

def get_datasets(dset_name):
    dset_name = dset_name.lower()
    print(f"Loading dataset ({dset_name})")
    if dset_name == 'mnist':
        train_dataset = dsets.MNIST(
            root='./data',
            train=True,
            transform=transforms.ToTensor(),
            download=True)
        new_train_dataset = []
        for d in train_dataset:
            new_train_dataset.append((d[0].numpy().flatten(), d[1]))
        train_dataset = new_train_dataset
        test_dataset = dsets.MNIST(
            root='./data',
            train=False,
            transform=transforms.ToTensor())
        new_test_dataset = []
        for d in test_dataset:
            new_test_dataset.append((d[0].numpy().flatten(), d[1]))
        test_dataset = new_test_dataset
    else:
        train_dataset, test_dataset = tabular_tools.get_dataset(dset_name)
    return train_dataset, test_dataset

def create_models(datasets, unit_inputs, unit_entries, unit_hashes, bits_per_input, num_workers, save_prefix="model"):
    # train_dataset, test_dataset = get_datasets(dset_name)

    # datasets = binarize_datasets(train_dataset, test_dataset, bits_per_input)
    prod = list(itertools.product(unit_inputs, unit_entries, unit_hashes)) 
    configurations = [datasets + c for c in prod]

    
    if num_workers == -1:
        num_workers = cpu_count()
    # print(f"Launching jobs for {len(configurations)} configurations across {num_workers} workers")
    with Pool(num_workers) as p:
        results = p.starmap(parameterized_run, configurations)
    # for entries in unit_entries:
    #     print(
    #         f"Best with {entries} entries: {max([results[i][1] for i in range(len(results)) if configurations[i][7] == entries])}")
    configs_plus_results = [[configurations[i][6:9]] +
                            list(results[i]) for i in range(len(results))]
    configs_plus_results.sort(reverse=True, key=lambda x: x[2])
    # for i in configs_plus_results:
        # print(f"{i[0]}: {i[2]} ({i[2] / len(datasets[4])})")

    return configs_plus_results[0]
    # Ensure folder for dataset exists
    # os.makedirs(os.path.dirname(f"./models/{dset_name}/{save_prefix}"), exist_ok=True)

    # for idx, result in enumerate(results):
    #     model = result[0]
    #     model_inputs, model_entries, model_hashes = configurations[idx][6:9]
    #     save_model(model, (datasets[0][0].size // bits_per_input),
    #         f"./models/{dset_name}/{save_prefix}_{model_inputs}input_{model_entries}entry_{model_hashes}hash_{bits_per_input}bpi.pickle.lzma")

def save_model(model, num_inputs, fname):
    model.binarize()
    model_info = {
        "num_inputs": num_inputs,
        "num_classes": len(model.discriminators),
        "bits_per_input": len(model.input_order) // num_inputs,
        "num_filter_inputs": model.discriminators[0].filters[0].num_inputs,
        "num_filter_entries": model.discriminators[0].filters[0].num_entries,
        "num_filter_hashes": model.discriminators[0].filters[0].num_hashes,\
        "hash_values": model.discriminators[0].filters[0].hash_values
    }
    state_dict = {
        "info": model_info,
        "model": model
    }

    with lzma.open(fname, "wb") as f:
        pickle.dump(state_dict, f)

def read_arguments():
    parser = argparse.ArgumentParser(description="Train BTHOWeN models for a dataset with specified hyperparameter sweep")
    parser.add_argument("dset_name", help="Name of dataset to use")
    parser.add_argument("--bits_per_input", nargs="+", required=True, type=int,  help="Number of thermometer encoding bits for each input in the dataset")
    args = parser.parse_args()
    return args

def random_chromosome():
    return {
        'filter_inputs': random.randint(8, 128),
        'filter_entries': 2 ** random.randint(2, 12),
        'filter_hashes': random.randint(1,8)
    }

def crossover(p1, p2):
    return {
        k: random.choice([p1[k], p2[k]]) for k in p1
    }

def mutate(chrom):
    if random.random() < MUTATION_RATE:
        chrom['filter_inputs'] = random.randint(8, 128)
    if random.random() < MUTATION_RATE:
        chrom['filter_entries'] = 2 ** random.randint(2, 12)
    if random.random() < MUTATION_RATE:
        chrom['filter_hashes'] = random.randint(1, 8)
    return chrom

def evaluate_chromosome(chrom, datasets, bpi, num_workers, save_prefix):
    return (
        chrom,
        create_models(
            datasets,
            [chrom['filter_inputs']],
            [chrom['filter_entries']],
            [chrom['filter_hashes']],
            bpi,
            num_workers,
            save_prefix
        )
    )

def evaluate_wrapper(args):
    return evaluate_chromosome(*args)

def main():
    args = read_arguments()

    save_prefix="model",
    num_workers= cpu_count() - 3

    for bpi in args.bits_per_input:
        train_dataset, test_dataset = get_datasets(args.dset_name)
        datasets = binarize_datasets(train_dataset, test_dataset, bpi)

        population = [random_chromosome() for _ in range(POPULATION_SIZE)]
        best_conf_model = None
        stagnant_generations = 0  # Counter for generations with no improvement

        print(f"Initial Population: {population}")

        for generation in range(GENERATIONS):
            
            print(f"Generation {generation + 1}/{GENERATIONS}")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                list_population_conf = list(executor.map(
                    evaluate_wrapper,
                    [(chrom, datasets, bpi, num_workers, save_prefix) for chrom in population]
                    ))
            
            list_population_conf.sort(key=lambda x: x[1][2], reverse=True)

            print(f"Best Chromosome of Generation {generation + 1}: {list_population_conf[0][1][2] }")
            
            current_best_conf_model = list_population_conf[0][1][0]

            if current_best_conf_model == best_conf_model:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
                best_conf_model = current_best_conf_model

            if stagnant_generations > 3:
                print("Stopping early due to no improvement in the last 3 generations.")
                break

            survivors = [chrom for chrom, _ in list_population_conf[:POPULATION_SIZE // 2]]

            children = []
            while len(children) < POPULATION_SIZE - len(survivors):
                p1, p2 = random.sample(survivors, 2)  # Generate two random parents from the survivors
                child = mutate(crossover(p1, p2))
                children.append(child)

            population = survivors + children

    if GENERATIONS > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list_final_population = list(executor.map(
                evaluate_wrapper,
                [(chrom, datasets, bpi, num_workers, save_prefix) for chrom in population]
            ))
    else: 
        list_final_population = list_population_conf

    list_final_population.sort(key = lambda x: x[1][2],reverse = True)
    best_score = list_final_population[0][1][2]
    best_model = list_final_population[0][1][1]
    best_model_inputs = list_final_population[0][1][0][0]
    best_model_entries = list_final_population[0][1][0][1]
    best_model_hashes = list_final_population[0][1][0][2]
    bpi = args.bits_per_input[0]

    # registrar melhor input 
    header = f"{'Score':<10}{'Dataset':<15}{'filter Inputs':<15}{'filter Entries':<15}{'filter Hashes':<15}{'bpi':<6}{'POPULATION_SIZE':<20}{'GENERATIONS':<15}{'MUTATION_RATE':<15}\n"
    with open("best_model.txt", "a+") as f:
        f.seek(0)
        content = f.read()
        if header not in content:
            f.write(f"\n{header}")
            f.write(f"{'-'*130}\n")
        f.write(f"{best_score:<10}{args.dset_name:<15}{best_model_inputs:<15}{best_model_entries:<15}{best_model_hashes:<15}{bpi:<6}{POPULATION_SIZE:<20}{GENERATIONS:<15}{MUTATION_RATE:<15}\n")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

