"""
    AbstractInstance

Abstract supertype for all instance types in the system.
An instance represents a single problem to be solved.
"""
abstract type AbstractInstance end

"""
    AbstractDataset

Abstract supertype for collections of instances.
A dataset provides a consistent interface for accessing multiple instances.
"""
abstract type AbstractDataset end

"""
    AbstractMethod

Abstract supertype for all solution methods.
Methods implement algorithms to solve instances.
"""
abstract type AbstractMethod end

"""
    AbstractMetric

Abstract supertype for metrics to evaluate solutions.
"""
abstract type AbstractMetric end

"""
    AbstractOutputFormat

Abstract supertype for output formatting strategies.
"""
abstract type AbstractOutputFormat end
