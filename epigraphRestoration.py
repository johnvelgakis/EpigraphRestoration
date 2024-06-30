import numpy as np
import pandas as pd
import os
import argparse
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deap import base, creator, tools
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load epigraph data
def load_epigraph_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    print('Data loaded successfully!')
    return df

# Filter epigraphs by geographic region
def filter_epigraphs_by_region(df, region_id):
    return df[df['region_main_id'] == region_id]

class GeneticAlgorithm:
    def __init__(self, epigraphs, target_epigraph, population_size, generations, crossover_rate, mutation_rate, elite_size, num_runs, improveThresh, stagThresh):
        self.epigraphs = epigraphs
        self.target_epigraph = target_epigraph
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.num_runs = num_runs
        self.improveThresh = improveThresh
        self.stagThresh = stagThresh
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), 
                                          max_features=1678,
                                          analyzer='word',
                                          norm='l2')
        self.transformed_epigraphs, self.tokens, self.vocabulary_dict = self.vectorize_texts(epigraphs)
        self.target_vector = self.vectorizer.transform([target_epigraph])
        self.dictionary_size = len(self.tokens)
        self.toolbox = base.Toolbox()
        self.setup_deap()

    def vectorize_texts(self, texts):
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        tokens = self.vectorizer.get_feature_names_out()
        vocabulary = {word: idx + 1 for idx, word in enumerate(tokens)}
        return tfidf_matrix, tokens, vocabulary

    def create_individual(self):
        word_indices = random.sample(range(1, self.dictionary_size + 1), 2)
        return creator.Individual(word_indices)

    def decode_individual(self, individual):
        reverse_vocabulary_dict = {v: k for k, v in self.vocabulary_dict.items()}
        words = [reverse_vocabulary_dict[idx] for idx in individual]
        return f" {words[0]} αλεξανδρε ουδις {words[1]}"

    def fitness_function(self, individual):
        completed_epigraph = self.decode_individual(individual)
        completed_vector = self.vectorizer.transform([completed_epigraph])
        similarity = cosine_similarity(self.target_vector, completed_vector)
        fitness = similarity[0][0]
        print(f"Individual: {individual}, Completed Epigraph: {completed_epigraph}, Target Vector: {self.target_vector}, Completed Vector: {completed_vector}, Similarity: {similarity}, Fitness: {fitness}") 
        return (fitness,)  

    def setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run_single(self):
        population = self.toolbox.population(n=self.population_size)
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        best_fitnesses = []
        stagnation_count = 0
        improvement_threshold = self.improveThresh
        max_generations = self.generations
        best_fitness = max(fitnesses)[0]

        for g in range(max_generations):
            if best_fitness >= 100:
                break

            print(f"-----------------\n Generation {g} \n-------------------")
            
            elite = tools.selBest(population, self.elite_size)
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))
            print(f'#Elite: {len(elite)}, #offspring: {len(offspring)}')
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = elite + offspring

            fits = [ind.fitness.values[0] for ind in population]
            current_best_fitness = max(fits)
            best_fitnesses.append(current_best_fitness)
            print(f"Best fitness of current generation: {current_best_fitness}")

            if g > 0 and (current_best_fitness - best_fitness) < improvement_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0

            best_fitness = current_best_fitness
            stagnation_threshold = self.stagThresh
            if stagnation_count >= stagnation_threshold:
                print("Termination: Stagnation limit reached")
                break

        best_individual = tools.selBest(population, 1)[0]
        best_epigraph = self.decode_individual(best_individual)
        print(f'Best individual: {best_individual} --> {best_epigraph},\nBest fitness: {best_fitness}')
        return best_individual, best_epigraph, best_fitnesses

    def run(self):
        best_individuals = []
        num_generations = []
        avg_fitness_per_gen = []
        best_epigraph_overall = None

        for _ in range(self.num_runs):
            best_individual, best_epigraph, best_fitness = self.run_single()
            best_individuals.append(best_individual.fitness.values[0])
            num_generations.append(len(best_fitness))

            if len(avg_fitness_per_gen) == 0:
                avg_fitness_per_gen = best_fitness
            else:
                avg_fitness_per_gen = [sum(x) / 2 for x in zip(avg_fitness_per_gen, best_fitness)]
            
            best_epigraph_overall = best_epigraph

        ave_best_fitness = np.mean(best_individuals)
        ave_generations = np.mean(num_generations)

        print("-----------------\n End of (successful) evolution \n-------------------")
        print("Average best fitness over all runs:", ave_best_fitness)
        print("Average number of generations:", ave_generations)

        trial_number = self.get_last_trial_number() + 1
        plt.plot(avg_fitness_per_gen)
        plt.xlabel('Generation')
        plt.ylabel('Average Best Fitness')
        plt.title('Evolution of Average Best Fitness Over Generations')
        if not os.path.exists('media'):
            os.makedirs('media')
        plt.savefig(f'media/trial{trial_number}.png')
        plt.show()
        print(self.target_vector)
        self.update_experiment_results(trial_number, best_epigraph_overall, ave_best_fitness, ave_generations)
        return best_epigraph_overall
    
    def get_last_trial_number(self):
        last_trial = 0
        if os.path.exists('experimentResults.txt'):
            with open('experimentResults.txt', 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1]
                    last_trial = int(last_line.split(":")[0].split("Trial")[1].strip())

        # if os.path.exists('experimentResults.csv'):
        #     df = pd.read_csv('experimentResults.csv')
        #     if not df.empty:
        #         last_trial = max(last_trial, df['trial'].max())

        return last_trial


    def update_experiment_results(self, trial_number, best_epigraph, ave_best_fitness, ave_generations):
        print(f"Updating experimentResults.txt with trial {trial_number}")
        # Update TXT
        with open('experimentResults.txt', 'a') as file:
            file.write(f"Trial {trial_number}: '{best_epigraph}' | num_runs={self.num_runs}, population_size={self.population_size}, generations={self.generations}, crossover_rate={self.crossover_rate}, mutation_rate={self.mutation_rate}, elite_size={self.elite_size}, improveThresh={self.improveThresh}, stagThresh={self.stagThresh} ---> ave_fitness={ave_best_fitness}, ave_generations={ave_generations}\n")
        print("Experiment results TXT updated successfully!")
        # Update CSV
        csv_file = 'experimentResults.csv'
        write_header = not os.path.exists(csv_file)
        with open(csv_file, 'a') as csvfile:
            fieldnames = ['trial', 'epigraphRestored', 'ave_fitness', 'ave_generations', 'num_runs', 'population_size', 'generations', 'crossover_rate', 'mutation_rate', 'elite_size', 'improveThresh', 'stagThresh']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'trial': trial_number,
                'epigraphRestored': best_epigraph,
                'ave_fitness': ave_best_fitness,
                'ave_generations': ave_generations,
                'num_runs': self.num_runs,
                'population_size': self.population_size,
                'generations': self.generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'elite_size': self.elite_size,
                'improveThresh': self.improveThresh,
                'stagThresh': self.stagThresh
            })
        print("Experiment results CSV updated successfully!")



# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Epigraph Restoration with Genetic Algorithm')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--population_size', type=int, help='Population size')
    parser.add_argument('--generations', type=int, help='Number of generations')
    parser.add_argument('--crossover_rate', type=float, help='Crossover rate')
    parser.add_argument('--mutation_rate', type=float, help='Mutation rate')
    parser.add_argument('--elite_size', type=int, help='Elite size')
    parser.add_argument('--num_runs', type=int, help='Number of runs')
    parser.add_argument('--improveThresh', type=float, help='Improvement threshold')
    parser.add_argument('--stagThresh', type=int, help='Stagnation threshold')

    args = parser.parse_args()

    file_path = 'data.csv'
    df = load_epigraph_data(file_path)
    filtered_epigraphs = filter_epigraphs_by_region(df, 1693)
    epigraph_texts = filtered_epigraphs['text'].tolist()
    target_epigraph = "αλεξανδρε ουδις"
    # For INTERACTIVE mode
    if args.interactive:
        print('-----------------\nEpigraph Restoration\n-----------------')
        print('Please enter the experiment\'s hyperparameters:')
        population_size = int(input("Population size: "))
        generations = int(input("Number of generations: "))
        crossover_rate = float(input("Crossover rate: "))
        mutation_rate = float(input("Mutation rate: "))
        elite_size = int(input("Elite size: "))
        num_runs = int(input("Number of runs: "))
        improveThresh = float(input("Improvement threshold: "))
        stagThresh = int(input("Stagnation threshold (generations): "))
    else:
        population_size = args.population_size or 100
        generations = args.generations or 1000
        crossover_rate = args.crossover_rate or 0.6
        mutation_rate = args.mutation_rate or 0.01
        elite_size = args.elite_size or 1
        num_runs = args.num_runs or 10
        improveThresh = args.improveThresh or 0.01
        stagThresh = args.stagThresh or 25

    kwargs = {'epigraphs': epigraph_texts, 
              'target_epigraph': target_epigraph, 
              'population_size': population_size, 
              'generations': generations, 
              'crossover_rate': crossover_rate, 
              'mutation_rate': mutation_rate, 
              'elite_size': elite_size, 
              'num_runs': num_runs, 
              'improveThresh': improveThresh, 
              'stagThresh': stagThresh}
    
    ga = GeneticAlgorithm(**kwargs)
    best_epigraph = ga.run()
    print("Keyword arguments to GeneticAlgorithm:", kwargs)
    print("\nBest Epigraph:", best_epigraph)
