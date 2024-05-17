import mesa
from model import DepressionSupportModel
from mesa.batchrunner import batch_run
import pandas as pd

def main():
    parameters = {
        "num_agents": range(10, 51, 10),
        "num_supportive": range(10, 51, 10),
        "stigma_level": [0.0, 0.25, 0.5, 0.75, 1.0],
        "support_effectiveness": [0.3, 0.5, 0.7]
    }

    results = batch_run(
        model_cls=DepressionSupportModel,
        parameters=parameters,
        iterations=5,
        max_steps=100,
        number_processes=4,
        data_collection_period=1,
        display_progress=True
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv("batch_run_results.csv")

if __name__ == '__main__':
    main()
