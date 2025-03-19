"""
    write_solution(solution::AbstractSolution, format::AbstractOutputFormat, path::String)

Write a solution to the specified path using the given format.
"""
function write_solution(solution::AbstractSolution, format::AbstractOutputFormat, path::String)
        error("write_solution not implemented for solution type $(typeof(solution)) and format $(typeof(format))")
end

"""
    write_summary(format::AbstractOutputFormat, directory::String, output_file::String; 
                 metric::Symbol, num_runs::Int=5)

Create a summary file by compiling results for a specific metric from solution files in the directory.
"""
function write_summary(format::AbstractOutputFormat, directory::String, output_file::String;
        metric::Symbol, num_runs::Int=5)
        error("write_summary not implemented for format $(typeof(format))")
end

# Export writer functions
export write_solution, write_summary
