
plot_contour(hours, Zprofile, Tprofile, t_plot, filename) = nothing
export_plot(all_data, ϕ_all, filename) = nothing
export_plot_snapshot(all_data, ϕ_all, filename) = nothing

# using Requires
# @init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
#   using .Plots


function plot_contour(hours, Zprofile, Tprofile, t_plot, filename)
    contour(
        hours,
        Zprofile .* 100,
        Tprofile,
        levels = 243.15:323.15,
        xticks = 0:12:t_plot,
        xlimits = (12, t_plot),
        xlabel = "Time of day (hours)",
        ylabel = "Soil depth (cm)",
        title = "Soil temperature (°K)",
    )
    savefig(filename)
end

function plot_friendly_name(ϕ)
    return replace(ϕ, "ρ" => "rho")
end

function export_plot(all_data, ϕ_all, filename)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    p = plot()
    z = all_data[1]["z"][:]
    for n in 0:(length(keys(all_data)) - 1)
        for ϕ in ϕ_all
            ϕ_string = String(ϕ)
            ϕ_name = plot_friendly_name(ϕ_string)
            ϕ_data = all_data[n][ϕ_string][:]
            plot!(ϕ_data, z, xlabel = ϕ_name, ylabel = "z [cm]")
        end
    end
    savefig(filename)
end

function export_plot_snapshot(all_data, ϕ_all, filename)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    p = plot()
    z = all_data["z"][:]
    for ϕ in ϕ_all
        ϕ_string = String(ϕ)
        ϕ_name = plot_friendly_name(ϕ_string)
        ϕ_data = all_data[ϕ_string][:]
        plot!(ϕ_data, z, xlabel = ϕ_name, ylabel = "z [cm]")
    end
    savefig(filename)
end

# end
