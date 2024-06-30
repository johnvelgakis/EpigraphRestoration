#  Epigraph Restoration Using Genetic Algorithms

 This project focuses on restoring ancient epigraphs using genetic algorithms. The core idea is to predict missing words in partially destroyed epigraphs based on the content of epigraphs from the same geographic region.

 ## Installation

 1. Clone the repository:
    ```sh
    git clone https://github.com/johnvelgakis/EpigraphRestoration.git
    cd EpigraphRestoration
    ```

 2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

 ## Usage

 To run the epigraph restoration process, use the following command:
 ```sh
 python epigraphRestoration.py --population_size 100 --generations 1000 --crossover_rate 0.6 --mutation_rate 0.01 --elite_size 1 --num_runs 10 --improveThresh 0.01 --stagThresh 25
 ```

 For interactive mode:
 ```sh
 python epigraphRestoration.py --interactive
 ```

 To run <number> experiments using random parameters:
 ```sh
 python testing.py --num_trials <number>
 ```

 To run the predefined experiments:
 ```sh
 python testing.py --exercise
 ```
 
 To view stored results:
 ```sh
 python testing.py --results
 ```

 ## Project Structure

 - **epigraphRestoration.py**: Main script for running the genetic algorithm for epigraph restoration.
 - **testing.py**: Script for running multiple experiments and viewing results.
 - **experimentResults.csv**: CSV file where experiment results are stored.
 - **experimentResults.txt**: Text file where experiment results are logged.
 - **data.csv**: Contains the epigraph data used for training and testing.
 - **media/**: Directory where generated plots are saved.

 ## Directory Structure

 ```
 EpigraphRestoration/
 │
 ├── epigraphRestoration.py
 ├── testing.py
 ├── experimentResults.csv
 ├── experimentResults.txt
 ├── data.csv
 ├── requirements.txt
 ├── LICENSE
 ├── README.md
 ├── media/
 │   └── (All the plots being saved)
 ```

 ## Examples

 Here are is an example of a restored epigraph:

 1. Original: [...] αλεξανδρε ουδις [...]
    Restored: αγενειων αλεξανδρε ουδις αρτεμισιου

 ## Contributing

 Contributions are welcome! Please follow these steps:

 1. Fork the repository.
 2. Create a new branch: `git checkout -b feature-branch`.
 3. Make your changes and commit them: `git commit -m 'Add new feature'`.
 4. Push to the branch: `git push origin feature-branch`.
 5. Submit a pull request.

 ## License

 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
 ## Contact

 For any questions or issues, please contact:

 - **Ioannis Velgakis** - (johnvelgakis@gmail.com)
