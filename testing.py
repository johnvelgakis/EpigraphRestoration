import subprocess
import random
import argparse
import tkinter as tk
from tkinter import Scrollbar, Canvas, Frame, Label, ttk, Text
from PIL import Image, ImageTk
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from time import time

def run_experiment(kwargs, start_trial):
    command = [
        'python', 'epigraphRestoration.py',
        '--population_size', str(kwargs['population_size']),
        '--generations', str(kwargs['generations']),
        '--crossover_rate', str(kwargs['crossover_rate']),
        '--mutation_rate', str(kwargs['mutation_rate']),
        '--elite_size', str(kwargs['elite_size']),
        '--num_runs', str(kwargs['num_runs']),
        '--improveThresh', str(kwargs['improveThresh']),
        '--stagThresh', str(kwargs['stagThresh'])
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    output_lines = result.stdout.split('\n')
    best_epigraph = None
    ave_fitness = None
    ave_generations = None

    for line in output_lines:
        if line.startswith("Best Epigraph:"):
            best_epigraph = line.split(":", 1)[1].strip()
        elif line.startswith("Average best fitness over all runs:"):
            ave_fitness = float(line.split(":", 1)[1].strip())
        elif line.startswith("Average number of generations:"):
            ave_generations = float(line.split(":", 1)[1].strip())

    return {
        'trial': start_trial,
        'best_epigraph': best_epigraph,
        'ave_fitness': ave_fitness,
        'ave_generations': ave_generations,
        'population_size': kwargs['population_size'],
        'generations': kwargs['generations'],
        'crossover_rate': kwargs['crossover_rate'],
        'mutation_rate': kwargs['mutation_rate'],
        'elite_size': kwargs['elite_size'],
        'num_runs': kwargs['num_runs'],
        'improveThresh': kwargs['improveThresh'],
        'stagThresh': kwargs['stagThresh']
    }

class FigureWindow:
    def __init__(self, num_trials):
        self.root = tk.Toplevel()
        self.root.title("Evolution of Average Best Fitness Over Generations")
        self.num_trials = num_trials

        if num_trials == 1:
            rows, cols = 1, 1
        elif num_trials <= 2:
            rows, cols = 1, 2
        elif num_trials <= 3:
            rows, cols = 1, 3
        elif num_trials <= 4:
            rows, cols = 2, 2
        elif num_trials <= 5:
            rows, cols = 1, 5
        elif num_trials <= 6:
            rows, cols = 2, 3
        elif num_trials <= 7:
            rows, cols = 3, 3
        else:
            rows, cols = (num_trials // 4) + 1, 4

        self.canvas = Canvas(self.root)
        self.scrollbar_v = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollbar_h = Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_v.set, xscrollcommand=self.scrollbar_h.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar_v.pack(side="right", fill="y")
        self.scrollbar_h.pack(side="bottom", fill="x")

        self.scrollable_frame.bind_all("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

        self.frames = []
        for i in range(num_trials):
            frame = tk.Frame(self.scrollable_frame)
            frame.grid(row=i // cols, column=i % cols, padx=10, pady=10)
            self.frames.append(frame)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")

    def add_figure(self, trial_number):
        image_path = f'media/trial{trial_number}.png'
        if os.path.exists(image_path):
            img = Image.open(image_path).resize((400, 300))
            img = ImageTk.PhotoImage(img)
            label_img = tk.Label(self.frames[trial_number - 1], image=img)
            label_img.image = img
            label_img.pack()
            
            label_text = tk.Label(self.frames[trial_number - 1], text=f'Trial {trial_number}')
            label_text.pack()

def show_results(results):
    root = tk.Tk()
    root.title("Experiment Results")

    columns = ['trial', 'Restored_epigraph', 'ave_fitness', 'ave_generations', 
            'population_size', 'generations', 'crossover_rate', 
            'mutation_rate', 'elite_size', 'num_runs', 
            'improveThresh', 'stagThresh']
    
    tree = ttk.Treeview(root, columns=columns, show='headings')
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    tree.pack(fill=tk.BOTH, expand=True)

    df = pd.DataFrame(results)
    df.columns = df.columns.str.strip()  # Strip leading and trailing spaces from column names

    # Debugging output for the DataFrame
    print(df.head())
    print(df.columns)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure 'ave_fitness' is numeric
    if 'ave_fitness' in df.columns:
        try:
            df['ave_fitness'] = pd.to_numeric(df['ave_fitness'], errors='coerce')
        except Exception as e:
            print(f"Error converting 'ave_fitness' to numeric: {e}")
            print(df['ave_fitness'].head())

    for i, row in df.iterrows():
        trial_data = (row['trial'], row.get('epigraphRestored', 'N/A'), row.get('ave_fitness', 'N/A'), row.get('ave_generations', 'N/A'),
                    row.get('population_size', 'N/A'), row.get('generations', 'N/A'), row.get('crossover_rate', 'N/A'), 
                    row.get('mutation_rate', 'N/A'), row.get('elite_size', 'N/A'), row.get('num_runs', 'N/A'), 
                    row.get('improveThresh', 'N/A'), row.get('stagThresh', 'N/A'))
        tree.insert("", tk.END, values=trial_data)

    # Find the epigraph with the highest average fitness
    best_epigraph = df.loc[df['ave_fitness'].idxmax()]['epigraphRestored'] if 'ave_fitness' in df.columns else 'N/A'

    # Display the epigraph with the highest average fitness
    best_epigraph_frame = tk.Frame(root)
    best_epigraph_frame.pack(fill=tk.BOTH, expand=True)

    best_epigraph_label = tk.Label(best_epigraph_frame, text="Epigraph with Highest Average Fitness:")
    best_epigraph_label.pack()

    best_epigraph_text = Text(best_epigraph_frame, height=5, width=100)
    best_epigraph_text.insert(tk.END, best_epigraph)
    best_epigraph_text.pack()

    # Create a frame for the second table and graph
    frame_top10 = tk.Frame(root)
    frame_top10.pack(fill=tk.BOTH, expand=True)

    # Get top 10 trials based on ave_fitness
    top10_df = df.nlargest(10, 'ave_fitness') if 'ave_fitness' in df.columns else df

    # Create a second treeview for top 10 trials
    top10_tree = ttk.Treeview(frame_top10, columns=columns, show='headings')
    for col in columns:
        top10_tree.heading(col, text=col)
        top10_tree.column(col, anchor='center')
    top10_tree.pack(fill=tk.BOTH, expand=True)
    
    for i, row in top10_df.iterrows():
        trial_data = (row['trial'], row.get('epigraphRestored', 'N/A'), row.get('ave_fitness', 'N/A'), row.get('ave_generations', 'N/A'),
                    row.get('population_size', 'N/A'), row.get('generations', 'N/A'), row.get('crossover_rate', 'N/A'), 
                    row.get('mutation_rate', 'N/A'), row.get('elite_size', 'N/A'), row.get('num_runs', 'N/A'), 
                    row.get('improveThresh', 'N/A'), row.get('stagThresh', 'N/A'))
        top10_tree.insert("", tk.END, values=trial_data)

    # Create a figure for the graph
    fig, ax = plt.subplots(figsize=(10, 5))
    if not top10_df.empty and 'ave_fitness' in top10_df.columns:
        bars = top10_df.plot(kind='bar', x='trial', y='ave_fitness', ax=ax)
        ax.set_title('Top 10 Trials by Average Fitness')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Average Fitness')

        # Add labels on top of each bar
        for bar in bars.patches:
            ax.annotate(f'{bar.get_height():.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()


def get_last_trial_number():
    last_trial = 0
    if os.path.exists('experimentResults.txt'):
        with open('experimentResults.txt', 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                last_trial = int(last_line.split(":")[0].split("Trial")[1].strip())

    if os.path.exists('experimentResults.csv'):
        df = pd.read_csv('experimentResults.csv')
        if not df.empty:
            last_trial = max(last_trial, df['trial'].max())

    return last_trial

def main():
    time_start = time()
    parser = argparse.ArgumentParser(description='Run multiple experiments for Epigraph Restoration')
    parser.add_argument('--exercise', action='store_true', help='Run predefined set of experiments')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of trials to run')
    parser.add_argument('--results', action='store_true', help='Show stored results')
    args = parser.parse_args()
    
    num_trials = args.num_trials
    
    trials = []
    last_trial_number = get_last_trial_number()
    start_trial = last_trial_number + 1

    if args.results:
        if not os.path.exists('experimentResults.csv'):
            print("No results found.")
            return
        df = pd.read_csv('experimentResults.csv')
        results = df.to_dict('records')
        
        # Schedule the creation of both windows
        root = tk.Tk()
        root.withdraw()
        root.after(0, lambda: show_results(results))
        root.after(500, lambda: create_figure_window(len(results)))
        root.mainloop()
        return

    if args.exercise:
        trials = [
            {'population_size': 20, 'generations': 1000, 'crossover_rate': 0.6, 'mutation_rate': 0, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 20, 'generations': 1000, 'crossover_rate': 0.6, 'mutation_rate': 0.01, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 20, 'generations': 1000, 'crossover_rate': 0.6, 'mutation_rate': 0.1, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 20, 'generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.01, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 20, 'generations': 1000, 'crossover_rate': 0.1, 'mutation_rate': 0.01, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 200, 'generations': 1000, 'crossover_rate': 0.6, 'mutation_rate': 0, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 200, 'generations': 1000, 'crossover_rate': 0.6, 'mutation_rate': 0.01, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 200, 'generations': 1000, 'crossover_rate': 0.6, 'mutation_rate': 0.1, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 200, 'generations': 1000, 'crossover_rate': 0.9, 'mutation_rate': 0.01, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50},
            {'population_size': 200, 'generations': 1000, 'crossover_rate': 0.1, 'mutation_rate': 0.01, 'elite_size': 1, 'num_runs': 10, 'improveThresh': 0.01, 'stagThresh': 50}
        ]
    else:
        for _ in range(num_trials):
            trials.append({
                'population_size': random.choice([20, 50, 100, 200, 250, 500, 1000]),
                'generations': random.choice([250, 500, 1000]),
                'crossover_rate': random.choice([0.1, 0.3, 0.6, 0.9]),
                'mutation_rate': random.choice([0, 0.01, 0.05, 0.1]),
                'elite_size': random.choice([0, 1, 2, 5, 10]),
                'num_runs': random.choice([10, 15, 25]),
                'improveThresh': random.choice([0.001, 0.01, 0.1]),
                'stagThresh': random.choice([20, 50, 100])
            })

    results = []
    for i, trial in enumerate(trials):
        print(f"\nStarting trial {start_trial + i}")
        result = run_experiment(trial, start_trial + i)
        results.append(result)

    # Update the experimentResults.csv and experimentResults.txt files
    df_new_results = pd.DataFrame(results)
    if os.path.exists('experimentResults.csv'):
        df_existing = pd.read_csv('experimentResults.csv')
        df_combined = pd.concat([df_existing, df_new_results], ignore_index=True)
    else:
        df_combined = df_new_results

    df_combined.to_csv('experimentResults.csv', index=False)

    with open('experimentResults.txt', 'a') as file:
        for result in results:
            file.write(f"Trial {result['trial']}: '{result['best_epigraph']}' | num_runs={result['num_runs']}, population_size={result['population_size']}, generations={result['generations']}, crossover_rate={result['crossover_rate']}, mutation_rate={result['mutation_rate']}, elite_size={result['elite_size']}, improveThresh={result['improveThresh']}, stagThresh={result['stagThresh']} ---> ave_fitness={result['ave_fitness']}, ave_generations={result['ave_generations']}\n")

    # Schedule the creation of both windows
    root = tk.Tk()
    root.withdraw()
    root.after(0, lambda: show_results(results))
    root.after(500, lambda: create_figure_window(len(results)))
    print(f'Took {(time() - time_start) / 60:.2f} minutes!')
    root.mainloop()

def create_figure_window(num_trials):
    fig_window = FigureWindow(num_trials)
    for i in range(1, num_trials + 1):
        fig_window.add_figure(i)
    fig_window.root.mainloop()

if __name__ == "__main__":
    main()
