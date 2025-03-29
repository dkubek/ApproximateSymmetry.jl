struct MultipleSimulationInstance <: AbstractInstance
    id::String
    simulations::Vector{MatrixInstance}
    path::String
end


export MultipleSimulationInstance