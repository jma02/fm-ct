`fm-ct.py` is the main file. It has some cli arguments that are required.
`fm-ct.qs` is for slurm. 
`grid.txt` would be useful for grid search but I'm leaving that for later if needed.
`make_dataset.py` generates our dataset, if needed.
`sample.py` takes as input a trained velocity_field.pt and generates a sample.

It looks like `lr = .0001` is way too slow. Model just does not converge. 
`lr = .001` is good with batch size 64.


