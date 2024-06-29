import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deap import base, creator, tools
import random
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
        print(f"Individual: {individual}, Completed Epigraph: {completed_epigraph}, Target Vector: {self.target_vector}, Completed Vector: {completed_vector}, Similarity: {similarity}, Fitness: {fitness}")  # Detailed Debug statement
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

            print(f"-- Generation {g} --")
            
            elite = tools.selBest(population, self.elite_size)
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))
            print(f'Elite: {len(elite)}, offspring: {len(offspring)}')
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
        return best_individual, best_fitnesses

    def run(self):
        best_individuals = []
        num_generations = []
        avg_fitness_per_gen = []

        for _ in range(self.num_runs):
            best_individual, best_fitnesses = self.run_single()
            best_individuals.append(best_individual.fitness.values[0])
            num_generations.append(len(best_fitnesses))

            if len(avg_fitness_per_gen) == 0:
                avg_fitness_per_gen = best_fitnesses
            else:
                avg_fitness_per_gen = [sum(x) / 2 for x in zip(avg_fitness_per_gen, best_fitnesses)]

        ave_best_fitness = np.mean(best_individuals)
        ave_generations = np.mean(num_generations)

        print("-- End of (successful) evolution --")
        print("Average best fitness over all runs:", ave_best_fitness)
        print("Average number of generations:", ave_generations)

        plt.plot(avg_fitness_per_gen)
        plt.xlabel('Generation')
        plt.ylabel('Average Best Fitness')
        plt.title('Evolution of Average Best Fitness Over Generations')
        plt.show()
        print(self.target_vector)
        return self.decode_individual(tools.selBest(self.toolbox.population(n=self.population_size), 1)[0])

# Main function
if __name__ == "__main__":
    file_path = 'data.csv'
    df = load_epigraph_data(file_path)
    filtered_epigraphs = filter_epigraphs_by_region(df, 1693)
    epigraph_texts = filtered_epigraphs['text'].tolist()
    
    target_epigraph = "αλεξανδρε ουδις"

    kwargs = {'epigraphs': epigraph_texts, 
              'target_epigraph': target_epigraph, 
              'population_size': 1000, 
              'generations': 2000, 
              'crossover_rate': 0.2, 
              'mutation_rate': 0.05, 
              'elite_size': 10, 
              'num_runs': 1000, 
              'improveThresh': 0.001, 
              'stagThresh': 100}
    
    ga = GeneticAlgorithm(**kwargs)
    best_epigraph = ga.run()
    print("Keyword arguments to GeneticAlgorithm:", kwargs)
    print("\nBest Epigraph:", best_epigraph)
    
