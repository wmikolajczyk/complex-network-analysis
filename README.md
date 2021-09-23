# Complex network analysis
This is the code of my master's thesis

### Preparation
#### Real graphs
1. Download and prepare raw dataset - extract archive to directory with dataset name in raw_datasets folder (raw_datasets/dataset_name/...)
2. Run prepare_datasetname method to create weighted edge list, and attributes list as .csv files
3. Load dataset to graph

#### Generative graphs
1. Generate graph based on passed parameters 

### Pipeline
4. Attach graph attributes
5. Attach real attributes (only real graphs)
6. Convert graph to training dataframe
7. Preprocess graph training dataframe
8. Train model
9. Recreate by priority rank
10. Get original and recreated graph measurements
11. Make comparison
