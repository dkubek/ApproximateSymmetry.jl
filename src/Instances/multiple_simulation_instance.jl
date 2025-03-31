struct MultipleSimulationInstance <: Instances.AbstractInstance
	id::String
	simulations::Vector{Instances.AbstractInstance}
	metadata::Dict{Symbol, Any}

	function MultipleSimulationInstance(id::String, simulations::Vector{I}) where I <: Instances.AbstractInstance
		if isempty(simulations)
			throw(Common.ApproximateSymmetryError("MultipleSimulationInstance must contain at least one simulation"))
		end
		new(id, simulations, Dict{Symbol, Any}())
	end
end

export MultipleSimulationInstance
