### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 2c0280a2-2db4-11ee-28c7-bbfc86608671
begin
	using Plots
	
	function gen_road_angle(n, m)
	    xs, ys = range(0, 1, n), range(-1, 1, m)
	    road, angle = rand(n, m), rand(-1:0.01:1)
	    width = round(Int, 0.35m) 
	    f(x) = angle * x^2
	    for (i, x) in enumerate(xs)
	        j = round(Int, (1-m)/(-1-1) * (f(x) - (-1)) + 1 )
	        w = width - round(Int, i/5) 
	        for k in j-w:j+w
	            if 1 <= k <= n
	                road[i, k] = 0.5road[i, k]
	            end
	        end
	    end
	    return road, angle
	end
	
	anim = @animate for _ in 1:50
	    road, angle = gen_road_angle(100, 100)
	    heatmap(road, legend = false, title = "Norm. angle: $angle", c = :grayC)
	end
	
	gif(anim, "lanes.gif", fps = 3)
end

# ╔═╡ aff1cd5b-9ae2-4fb6-aa7a-47e4f16d4abf
begin
	using Pkg
	Pkg.add("DataFrames")
	using CSV
	using DataFrames

	function generate_dataset(num_imgs, length_pixels, width_pixels, filename)
		dataset = []
		for _ in 1:num_imgs
			road, angle = gen_road_angle(length_pixels, width_pixels)
			sample = (road, angle)
			push!(dataset, sample)
		end
		df = DataFrame(Image = [sample[1] for sample in dataset], Angle = [sample[2] for sample in dataset])
		CSV.write(filename, df)
	end
	generate_dataset(5000, 100, 100, "lane_dataset.csv")
end

# ╔═╡ Cell order:
# ╠═2c0280a2-2db4-11ee-28c7-bbfc86608671
# ╠═aff1cd5b-9ae2-4fb6-aa7a-47e4f16d4abf
