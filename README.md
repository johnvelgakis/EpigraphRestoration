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

 To run `<number>` experiments using random parameters from Table 1:
 ```sh
 python testing.py --num_trials <number>
 ```
<div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center;">
  <div>
    <strong>Table 1: Parameters</strong>
    <table border="1" style="width: 50%; margin: 0 auto;">
      <tr>
        <th>Parameters</th>
        <th>Values</th>
      </tr>
      <tr>
        <td>population size</td>
        <td>[20, 50, 100, 200, 250, 500, 1000]</td>
      </tr>
      <tr>
        <td>number of generations</td>
        <td>[250, 500, 1000]</td>
      </tr>
      <tr>
        <td>crossover probability</td>
        <td>[0.1, 0.3, 0.6, 0.9]</td>
      </tr>
      <tr>
        <td>mutation probability</td>
        <td>[0, 0.01, 0.05, 0.1]</td>
      </tr>
      <tr>
        <td>number of elits</td>
        <td>[0, 1, 2, 5, 10]</td>
      </tr>
      <tr>
        <td>number of runs</td>
        <td>[10, 15, 25]</td>
      </tr>
      <tr>
        <td>improve threshold</td>
        <td>[0.001, 0.01, 0.1]</td>
      </tr>
      <tr>
        <td>stagnation threshold</td>
        <td>[20, 50, 100]</td>
      </tr>
    </table>
  </div>
</div>





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

 Here is an example of a restored epigraph:

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
