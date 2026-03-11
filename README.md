## Data Preparation and Inference Pipeline

1. **Generate k-mer profiles**

   Use [Draven](https://github.com/FilipTomas/Draven) to produce per-read, per-position k-mer profiles.  
   The output should be `.TSV` files containing the following columns:

   - `k-mer multiplicity`
   - `k-mer hash`

2. **Assign ground truth labels**

   Run `scripts/data_preprocess.py` to assign ground-truth labels to each position in the generated profiles.

3. **Create training batches**

   Use `scripts/create_batches.py` to convert the labeled data into batched datasets suitable for model training or inference.

4. **Run inference**

   Finally, run `batched_inference.py` to evaluate the model and measure its performance on the prepared batches.
