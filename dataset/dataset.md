# Dataset Generation
To generate a new dataset: 
1. run `generate_setup.py`.  Call `python generate_setup --help` for a complete list of options.
2. Open the generated setup files in Wireless Insite.
3. Run Wireless Insite to generate RF data.
4. Call `insite_to_dataset_parallel.py` to generate the TF Datasets for use in the model (Coming Soon)