"""
Process all instances in a dataset using the specified method and save results to the output directory.
Uses the specified output format for writing results.
"""
function process_dataset(
        dataset::AbstractDataset,
        method::AbstractMethod,
        output_dir::String,
        format::AbstractOutputFormat;
        num_runs=5,
        force_recompute=false
)
        error("process_dataset not implemented for dataset type $(typeof(dataset))")
end

export process_dataset
