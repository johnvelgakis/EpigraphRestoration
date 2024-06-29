import subprocess
import random
import argparse
import tkinter as tk
from tkinter import Scrollbar, Canvas, Frame, Label, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def run_experiment(kwargs):
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
        img = Image.open(f'media/trial{trial_number}.png').resize((400, 300))
        img = ImageTk.PhotoImage(img)
        label_img = tk.Label(self.frames[trial_number - 1], image=img)
        label_img.image = img
        label_img.pack()
        
        label_text = tk.Label(self.frames[trial_number - 1], text=f'Trial {trial_number}')
        label_text.pack()

def show_results(results):
    root = tk.Tk()
    root.title("Experiment Results")

    columns = ['trial', 'epigraphRestored', 'ave_fitness', 'ave_generations', 
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
    
    # Add 'trial' column to the DataFrame
    df['trial'] = df.index + 1

    for i, row in df.iterrows():
        trial_data = (i + 1, row.get('epigraphRestored', 'N/A'), row.get('ave_fitness', 'N/A'), row.get('ave_generations', 'N/A'),
                      row.get('population_size', 'N/A'), row.get('generations', 'N/A'), row.get('crossover_rate', 'N/A'), 
                      row.get('mutation_rate', 'N/A'), row.get('elite_size', 'N/A'), row.get('num_runs', 'N/A'), 
                      row.get('improveThresh', 'N/A'), row.get('stagThresh', 'N/A'))
        tree.insert("", tk.END, values=trial_data)

    # Create a frame for the second table and graph
    frame_top10 = tk.Frame(root)
    frame_top10.pack(fill=tk.BOTH, expand=True)

    # Get top 10 trials based on ave_fitness
    top10_df = df.nlargest(10, 'ave_fitness')

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

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments for Epigraph Restoration')
    parser.add_argument('--exercise', action='store_true', help='Run predefined set of experiments')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--results', action='store_true', help='Show stored results')
    args = parser.parse_args()

    num_trials = args.num_trials
    trials = []

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
                'num_runs': random.choice([10, 25, 50, 100]),
                'improveThresh': random.choice([0.001, 0.01, 0.1]),
                'stagThresh': random.choice([20, 50, 100])
            })

    results = []
    for i, trial in enumerate(trials):
        print(f"\nStarting trial {i+1}")
        result = run_experiment(trial)
        results.append(result)

    # Schedule the creation of both windows
    root = tk.Tk()
    root.withdraw()
    root.after(0, lambda: show_results(results))
    root.after(500, lambda: create_figure_window(len(results)))
    root.mainloop()

def create_figure_window(num_trials):
    fig_window = FigureWindow(num_trials)
    for i in range(1, num_trials + 1):
        fig_window.add_figure(i)
    fig_window.root.mainloop()

if __name__ == "__main__":
    main()
