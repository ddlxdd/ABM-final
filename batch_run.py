import mesa
from model import DepressionSupportModel, compute_average_depression, count_state
from mesa.batchrunner import batch_run
import pandas as pd

def main():
    
    parameters = {
        "num_agents": [50],
        "num_supportive": [10],
        "stigma_level": [0.1, 0.3, 0.5, 0.7, 0.9], 
        "support_effectiveness": [0.1, 0.3, 0.5, 0.7, 0.9],
        "acceptance_probability": [0.3, 0.5, 0.7, 0.9],
        "refusal_gets_worse_probability": [0.3, 0.5, 0.7, 0.9]
    }

   
    results = batch_run(
        DepressionSupportModel,
        parameters,
        iterations=3, 
        max_steps=50, 
        number_processes=None,
        data_collection_period=-1,
        display_progress=True
    )

    
    results_df = pd.DataFrame(results)
    results_df.to_csv('batch_run_results_updated.csv')

if __name__ == "__main__":
    main()
