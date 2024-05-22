import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


batch_run_results_path = 'batch_run_results.csv'
batch_run_results = pd.read_csv(batch_run_results_path)


correlation_matrix = batch_run_results[['num_agents', 'num_supportive', 'stigma_level', 'support_effectiveness', 'acceptance_probability', 'refusal_gets_worse_probability', 'Average Depression', 'Mild Count', 'Moderate Count', 'Severe Count']].corr()


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()