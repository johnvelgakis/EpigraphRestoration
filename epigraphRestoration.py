import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import vectorize_texts
import random

# Load epigraph data
def load_epigraph_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Filter epigraphs by geographic region
def filter_epigraphs_by_region(df, region_id):
    return df[df['geo_location'] == region_id]

# Genetic Algorithm functions
def create_initial_population(dictionary_size, population_size):
    return [random.sample(range(1, dictionary_size + 1), 2) for _ in range(population_size)]

def fitness_function(individual, target_vector, dictionary, tokens):
    words = [tokens[i - 1] for i in individual]
    completed_epigraph = " ".join(words)
    completed_vector = dictionary.transform([completed_epigraph])
    similarity = cosine_similarity(target_vector, completed_vector)
    return similarity[0][0]

def selection(population, fitnesses):
    selected = random.choices(population, weights=fitnesses, k=len(population))
    return selected

def crossover(parent1, parent2):
    crossover_point = random.randint(0, 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, dictionary_size, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(1, dictionary_size)
    return individual

def genetic_algorithm(epigraphs, target_epigraph, population_size=100, generations=1000, mutation_rate=0.01):
    dictionary, tokens = vectorize_texts(epigraphs)
    target_vector = dictionary.transform([target_epigraph])
    dictionary_size = len(tokens)
    
    population = create_initial_population(dictionary_size, population_size)
    for generation in range(generations):
        fitnesses = [fitness_function(individual, target_vector, dictionary, tokens) for individual in population]
        
        if max(fitnesses) == 1.0:
            break
        
        population = selection(population, fitnesses)
        next_population = []
        
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([mutate(child1, dictionary_size, mutation_rate), mutate(child2, dictionary_size, mutation_rate)])
        
        population = next_population
    
    best_individual = population[np.argmax(fitnesses)]
    best_words = [tokens[i - 1] for i in best_individual]
    return best_words

# Main function
if __name__ == "__main__":
    file_path = 'iphi2802.csv'
    df = load_epigraph_data(file_path)
    filtered_epigraphs = filter_epigraphs_by_region(df, 1683)
    epigraph_texts = filtered_epigraphs['text'].tolist()
    
    target_epigraph = "[...] αλεξανδρε ουδις [...]"
    
    restored_words = genetic_algorithm(epigraph_texts, target_epigraph)
    print("Restored words:", restored_words)
